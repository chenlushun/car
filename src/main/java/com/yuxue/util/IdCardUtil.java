package com.yuxue.util;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.util.StringUtils;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.yuxue.constant.Constant;
import com.yuxue.entity.Line;
import com.yuxue.entity.LineClass;

import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import net.sourceforge.tess4j.util.LoadLibs;


/**
 * 证件识别工具类
 * 
 * @author yuxue
 * @date 2020-11-23 16:31
 */
public class IdCardUtil {

    private static final String TEMP_PATH = "D:/CardDetect/temp/";

    // 人脸识别库
    private static CascadeClassifier faceDetector;

    // Tess文字识别库
    private static Tesseract instance = new Tesseract();

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //设置tess4j配置的路径
        File testDataFolderFile = LoadLibs.extractTessResources("tessdata");
        // instance.setLanguage("eng");    // 加载语言模型 英文、数字；默认
        instance.setLanguage("chi_sim"); // 加载语言模型 中文、英文、数字
        // instance.setTessVariable("digits", "0123456789X");
        instance.setDatapath(testDataFolderFile.getAbsolutePath());
    }

    // 构造函数，加载默认模型文件
    IdCardUtil(){
        // faceDetector = new CascadeClassifier(Constant.DEFAULT_FACE_MODEL_PATH);
        faceDetector = new CascadeClassifier("D:\\CardDetect\\haarcascade_frontalface_default.xml");
    }

    // 加载自定义模型文件
    public void loadModel(String modelPath){
        if(!StringUtils.isEmpty(modelPath)) {
            faceDetector = new CascadeClassifier(modelPath);
        }
    }


    /**
     * 检测证件的人脸，获取人脸位置数据
     * @param grey 灰度图
     * @param debug
     * @param tempPath
     */
    public static Rect getFace(Mat grey, Boolean debug, String tempPath) {
        if(null == faceDetector || faceDetector.empty()) {
            System.out.println("加载模型文件失败: " + Constant.DEFAULT_FACE_MODEL_PATH);
            return null;
        }

        Rect dst = new Rect();
        MatOfRect faceDetected = new MatOfRect(); // 识别结果存储对象 // Rect矩形集合类
        faceDetector.detectMultiScale(grey, faceDetected); // 识别人脸
        Rect[] faceRect = faceDetected.toArray();
        if(faceRect.length > 0) {
            dst = faceRect[0]; // // 默认返回检测到的第一张人脸
            if(debug) {
                Mat m = grey.clone();
                for (Rect rect : faceRect) {
                    // 描绘边框
                    Imgproc.rectangle(m, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0));
                    // 输出图片
                    ImageUtil.debugImg(debug, tempPath, "getFace", m);
                }
            }
        }
        return dst;
    }



    /**
     * 霍夫线变换算法，提取直线，如果直线能围成一个矩形，则返回矩形的轮廓
     * 否则返回整个图片的边缘轮廓(即:假定整张图片都是证件内容)
     * @param threshold 边缘二值图像
     * @param debug
     * @param tempPath
     */
    public static Mat getCard(Mat grey, Mat threshold, Boolean debug, String tempPath) {
        // 检测结果 线段  // 4通道Mat，用于存储线段两个点的坐标，每一行一个像素点，代表一条线段
        Mat lines = new Mat();

        // 统计概率霍夫线变换；输出线段两个点的坐标
        // rho:就是一个半径的分辨率；theta:角度分辨率； threshold:判断直线点数的阈值
        // minLineLength：线段长度阈值；minLineGap:线段上最近两点之间的阈值
        Imgproc.HoughLinesP(threshold, lines, 1, Math.PI/180, 50, 100, 1); 

        if(lines.rows() <= 0) { // 没有扫描到直线
            return null;
        }

        if(debug) { // 将检测到的线段描绘出来
            Mat dst = threshold.clone();
            Imgproc.cvtColor(threshold, dst, Imgproc.COLOR_GRAY2BGR);
            Scalar scalar = new Scalar(0, 255, 0, 255); //蓝色
            for (int i = 0; i < lines.rows(); i++) {
                Line line = new Line(lines.row(i));
                Imgproc.line(dst, line.getStart(), line.getEnd(), scalar);
            }
            ImageUtil.debugImg(debug, tempPath, "drawLines", dst);
        }

        // HoughLinesP可能会将一条边框，扫描出来很多线段，而且还可能中间是断开的 
        // 将线段过滤并归归类合并；将重叠靠近的线段合并保留一条，并将中间断开的连接起来
        List<Line> ls = filterLines(threshold, lines, debug, tempPath);
        System.err.println("lines===>" + ls.size());

        // 用剩下的线段，围成矩形框; 
        // 如果是包含人脸的证件，可以使用人脸中心点的位置辅助计算
        // ？？是否还有必要提取顶点，进行投影校正？？
        List<Point> p = null;
        RotatedRect r = null;
        switch (ls.size()) {
        case 0: // 如果没有提取到有效的直线，则默认为整个图都是证件
            
            break;
        case 1: // 如果只有1条线段，默认就是边框线，按比例计算宽度或者高度，围成矩形
            r = getRectPoint(threshold, ls.get(0), debug, tempPath);
            break;
        case 2: // 如果只有2条线段，
            // 判断是垂直的，以交点为顶点，描绘矩形框，从而确定卡片矩形框 
            // ？？可以考虑是否进行角度校正；即：两条线段角度实际没有90°，校正到90°
            
            // 判断是平行的，提取四个顶点？

            r = getRectPoint(threshold, ls.get(0), debug, tempPath);
            break;

        default: 
            // 如果2条以上，按斜率再次分为两类， 得到较优的两条线段；多余的不要
            // 可以考虑， 计算线段延长线交点的位置，是否还在pic内部， 如果在的话，可以粗略判定都是卡片的边框
            r = getRectPoint(threshold, ls.get(0), debug, tempPath);
            break;
        }

        // 提取卡片切图   未完成 
        Mat card = new Mat(151 * 2, 254 * 2, grey.type());
        
        //
        
        return card;
    }


    /**
     * 根据斜率判断线段是卡片的底边还是侧边; 从而计算出卡片的宽度跟高度
     * 以线段的任一点为中心、宽、高、斜率确定一个斜矩形，其四个顶点必有一个是卡片的中心点
     * 如果有人脸位置信息，还可以继续确定具体是卡片的哪条边
     * @param line
     */
    public static RotatedRect getRectPoint(Mat threshold, Line line, Boolean debug, String tempPath) {
        double width = 0;
        double height = 0;
        double k = line.getK();
        Size size = null;
        if(Math.abs(k) < 1) { // 判定为底边
            width = line.getLength(); // 可以考虑弧度的影响，也可以不用考虑；因为弧度范围内没有需要识别的文字内容
            height = width / 1.58; // 身份证长宽比例约为1.58:1
            size = new Size(width, height);
        } else { // 判定为侧边
            height = line.getLength();
            width =  height * 1.58;
            size = new Size(height, width);
        }

        // 计算边相对X轴的角度
        double angle = Math.atan(k) / Math.PI * 180;
        /*System.out.println("width===>" + width);
        System.out.println("height===>" + height);
        System.out.println("k===>" + k);
        System.out.println("angle===>" + angle);*/

        // 以线段的任一点为中心、宽、高、斜率确定一个斜矩形，其四个顶点必有一个是卡片的中心点
        RotatedRect r0 = new RotatedRect(line.getStart(), size, angle);
        /*if(debug) { // 将检测到的线段描绘出来
            Mat dst = threshold.clone();
            Imgproc.cvtColor(threshold, dst, Imgproc.COLOR_GRAY2BGR);
            ImageUtil.drawRectangle(dst, r0);
            ImageUtil.debugImg(debug, tempPath, "drawRect", dst);
        }*/
        Point[] pt = new Point[4];
        r0.points(pt); // 获取到四个顶点

        Point center = null;
        double min = Double.MAX_VALUE;
        Point picCenter = new Point(threshold.width()/2, threshold.height()/2);
        for (Point p : pt) {
            // 直接计算距离图片中心点最近的顶点，即为卡片矩形中心点
            double d = ImageUtil.getDistance(p, picCenter);
            if(d < min) {
                min = d;
                center = p;
            }
        }
        RotatedRect r = new RotatedRect(center, size, angle);

        if(debug) {
            Mat dst = threshold.clone();
            Imgproc.cvtColor(threshold, dst, Imgproc.COLOR_GRAY2BGR);
            ImageUtil.drawRectangle(dst, r);
            Scalar scalar = new Scalar(0, 0, 255, 255); // 红色
            Imgproc.line(dst, line.getStart(), line.getEnd(), scalar);
            ImageUtil.debugImg(debug, tempPath, "drawRect", dst);
        }

        return r;
    }


    /**
     * 按照线段相对原点距离、斜率 进行分类 
     * 距离差值小于指定像素值，则判定为一类线段；即：这一类的线段，可能落在同一条直线线，可能都是是证件的边框线
     * 同类线段，按照最小起点，最大终点得到新的线段  //处理中间断开的线段
     * @param inMat
     * @param lines
     * @param debug
     * @param tempPath
     * @return
     */
    public static List<Line> filterLines(Mat threshold, Mat lines, Boolean debug, String tempPath){
        List<Line> result = Lists.newArrayList();
        List<LineClass> lineClass = Lists.newArrayList();// 按相对距离、斜率将线段分类
        for (int i = 1; i < lines.rows(); i++) {
            Line line = new Line(lines.row(i));
            boolean bl = false;
            for (LineClass lc : lineClass) {
                bl = lc.addLine(line);
                if(bl) { // 有满足的类
                    break;
                }
            }
            if(!bl) { // 没有满足的类，自己创建一个类
                lineClass.add(new LineClass(line)); 
            }
        }
        for (LineClass lc : lineClass) {
            result.add(lc.getNewLine());
        }
        if(debug) { // 将检测到的线段描绘出来
            Mat dst = threshold.clone();
            Imgproc.cvtColor(threshold, dst, Imgproc.COLOR_GRAY2BGR);
            Scalar scalar = new Scalar(0, 255, 0, 255); //蓝色
            for (Line line : result) {
                Imgproc.line(dst, line.getStart(), line.getEnd(), scalar);
            }
            ImageUtil.debugImg(debug, tempPath, "filterLines", dst);
        }
        return result;
    }



    /**
     * 筛选轮廓, 返回证件结果: 校正后的灰度图
     * 固定大小
     * @param inMat
     * @param face
     * @param contours
     * @param debug
     * @param tempPath
     */
    public static void getCard(Mat inMat, Mat dst, List<MatOfPoint> contours, Boolean debug, String tempPath) {
        // 根据人脸，预估证件的大小 // 非必须
        Double maxArea = inMat.width() * inMat.height() * 1.0;
        Double minArea =  inMat.width() * inMat.height() * 0.3; // 证件图像，至少占页面大小的1/3
        for (MatOfPoint c : contours) {
            // 获取最小外接矩形
            MatOfPoint2f mop2 = new MatOfPoint2f(c.toArray());
            RotatedRect rect = Imgproc.minAreaRect(mop2);

            // 验证尺寸
            if (minArea <= rect.size.area() && rect.size.area() <= maxArea) {
                if (debug) {
                    Mat d = inMat.clone();
                    ImageUtil.drawRectangle(d, rect);
                    ImageUtil.debugImg(debug, tempPath, "minAreaRect", d);
                }
                double angle = rect.angle;
                Size rect_size = new Size((int) rect.size.width, (int) rect.size.height);
                if (rect.size.width < rect.size.height) {
                    angle = 90 + angle;
                    rect_size = new Size(rect_size.height, rect_size.width);
                }
                // 根据人脸中心点位置，判断是否需要进行水平或者垂直180°旋转   // 不一定需要
                // 一般手机拍摄的照片，都是比较端正的，不需要进行水平或者垂直旋转，除非是故意的

                Mat clone = inMat.clone();
                // 校正图像
                shearCorrection(clone, clone, rect, mop2, debug, tempPath);

                // 旋转校正
                if(angle != 0) {
                    Mat rotmat = Imgproc.getRotationMatrix2D(rect.center, angle, 1);
                    Imgproc.warpAffine(clone, clone, rotmat, inMat.size());
                }

                // 裁剪
                Point p = new Point(rect.center.x, rect.center.y-2);
                Imgproc.getRectSubPix(clone, rect_size, p, clone);
                // 身份证大小:长85.6mm*宽54mm; 长度:240像素,高度:151像素
                Size dstSize = new Size(240 * 2, 151 * 2);
                Imgproc.resize(clone, dst, dstSize, 0, 0, Imgproc.INTER_CUBIC);
                ImageUtil.debugImg(debug, tempPath, "crop_resize", dst);
                break;
            }
        }
    }



    /**
     * 图像校正
     * 只能根据证件边框、最小外接矩形两个参数进行校正
     * 可以考虑按四个角最近点的来计算校正位置
     * @param rect 外接矩形
     * @param contour 证件轮廓
     * @param debug
     * @param tempPath
     * @return
     */
    public static void shearCorrection(Mat inMat, Mat dst, RotatedRect rect, MatOfPoint2f contour, Boolean debug, String tempPath){
        // 遍历轮廓的点，获取其四个顶点
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        // epsilon：输出精度，即两个轮廓点之间最大距离数 5,6,7... // closed：表示输出的多边形是否封闭
        Imgproc.approxPolyDP(contour, approxCurve, 10, false);
        Point[] points = approxCurve.toArray();
        if(points.length < 4) {
            return;
        } 
        // 提取轮廓四个顶点
        Point cp0 = null, cp1= null, cp2= null, cp3= null; 
        double maxSum = 0;
        double minSum = 1000000;
        for (Point p : points) {
            if(p.x + p.y >= maxSum) {    // xy求和最大，确定右下顶点
                cp0 = p;
                maxSum = p.x + p.y;
            }
            if(p.x + p.y <= minSum) { // xy求和最小，确定左上顶点
                cp3 = p;
                minSum = p.x + p.y;
            }
        }
        cp1 = getNearestPoint(points, new Point(cp3.x, cp0.y));
        cp2 = getNearestPoint(points, new Point(cp0.x, cp3.y));

        // 将四个顶点，跟最小外接矩形的顶点矩形进行匹配
        Mat vertex = new Mat(); 
        Imgproc.boxPoints(rect, vertex);  // 最小外接矩形，四个顶点 Mat(4, 2)
        Point rp0 = getNearestPoint(vertex, cp0);
        Point rp1 = getNearestPoint(vertex, cp1);
        Point rp2 = getNearestPoint(vertex, cp2);
        Point rp3 = getNearestPoint(vertex, cp3);

        // 投影变换 校正
        MatOfPoint2f srcPoints = new MatOfPoint2f(cp0, cp1, cp2, cp3);  // 原图四个顶点
        MatOfPoint2f dstPoints = new MatOfPoint2f(rp0, rp1, rp2, rp3);  // 目标图四个顶点
        Mat trans_mat  = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Imgproc.warpPerspective(inMat, dst, trans_mat, inMat.size());
        ImageUtil.debugImg(debug, tempPath, "warpPerspective", dst);
    }


    public static Point getNearestPoint(Point[] points, Point src) {
        double minDistance = 1000000;
        Point dst = null;
        for (Point p : points) {
            double d = ImageUtil.getDistance(p, src);
            if(d <= minDistance) {
                minDistance = d;
                dst = p; 
            }
        }
        return dst;
    }


    public static Point getNearestPoint(Mat vertex, Point src) {
        double minDistance = 1000000;
        Point dst = null;
        for (int i = 0; i < vertex.rows(); i++) {
            Point p = new Point(vertex.get(i, 0)[0], vertex.get(i, 1)[0]);
            double d = ImageUtil.getDistance(p, src);
            if(d <= minDistance) {
                minDistance = d;
                dst = p; 
            }
        }
        return dst;
    }


    public static BufferedImage Mat2BufImg(Mat matrix, String fileExtension) {
        MatOfByte  mob = new MatOfByte();
        Imgcodecs.imencode(fileExtension, matrix, mob);
        byte[] byteArray = mob.toArray();
        BufferedImage bufImage = null;
        try{
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bufImage;
    }


    public static Mat BufImg2Mat (BufferedImage original, int imgType, int matType) {
        if (original.getType() != imgType) {
            BufferedImage image = new BufferedImage(original.getWidth(), original.getHeight(), imgType);
            Graphics2D g = image.createGraphics();
            try {
                g.setComposite(AlphaComposite.Src);
                g.drawImage(original, 0, 0, null);
            } finally {
                g.dispose();
            }
        }
        byte[] pixels = ((DataBufferByte) original.getRaster().getDataBuffer()).getData();
        Mat mat = Mat.eye(original.getHeight(), original.getWidth(), matType);
        mat.put(0, 0, pixels);
        return mat;
    }


    /**
     * 使用tess4j识别字符
     * @param file 灰度图
     * @param r 字符区域
     * @return
     */
    public  static String recoChars(File file, Rect r) {

        /*java.awt.Rectangle rect = new Rectangle();
        rect.setRect(r.x, r.y, r.width, r.height);*/

        // 将验证码图片的内容识别为字符串
        String result = "";
        try {
            // result = instance.doOCR(file, rect); // 根据文件、框选的区域进行定向识别
            BufferedImage image = ImageIO.read(file); // 识别图片上所有文字

            // 识别图片上的所有文字
            result = instance.doOCR(image).replaceAll("%", "X").replaceAll(" ", "").replaceAll("\n", ""); 
            System.err.println("===>" + result);
        } catch (IOException | TesseractException e) {
            e.printStackTrace();
        }
        return result;
    }



    /**
     * 证件文字识别
     * 当前demo的应用场景：
     *      证件放置在桌子上，手机直接拍摄的照片；即:卡片位置不明确的图片，检测到卡片位置，并提取指定位置的文字信息；
     *      提取卡片位置的算法，是具有一定通用性的，不能兼顾所有的应用场景，通用性越高，相对应的成功率越低；
     *      在实际项目中，是可以通过其他手段获取到卡片位置信息的，比如：拍照或者上传图片的时候，显示一个矩形框，要求用户自行圈定卡片位置；
     * @param src
     * @param debug
     * @param tempPath
     */
    public static void cardDetect(Mat src, Boolean debug, String tempPath) {
        ImageUtil.debugImg(debug, tempPath, "src", src);
        Mat gsMat = new Mat();

        ImageUtil.GS_BLUR_KERNEL = 7;
        ImageUtil.gaussianBlur(src, gsMat, debug, tempPath);

        Mat grey = new Mat();
        ImageUtil.gray(gsMat, grey, debug, tempPath);

        // 检测到人脸位置 // 要求人脸检测算法比较精确 // 包含人脸的证件图片，可以用于提高定位的精确度
        Rect face = getFace(grey, debug, tempPath);
        // System.out.println("人脸中心点坐标===>" + face.x + "," + face.y);

        // 使用轮廓提取的方式获取证件位置，这里起决定性作用
        Mat scharr = new Mat();
        ImageUtil.scharr(grey, scharr, debug, tempPath);

        // 图像进行二值化
        Mat threshold = new Mat();
        ImageUtil.threshold(scharr, threshold, debug, tempPath);

        // 边缘腐蚀
        threshold = ImageUtil.erode(threshold, debug, tempPath, 2, 2);

        // 提取卡片Mat，方法一：
        // 提取二值图像的所有轮廓
        // List<MatOfPoint> contours = ImageUtil.contours(src, threshold, false, tempPath);
        //轮廓筛选, 获取最大的类矩形的轮廓;  即证件边框
        // Mat card = new Mat();
        // getCard(gsMat, card, contours, debug, tempPath);

        // 提取卡片Mat，方法二：
        // 霍夫线方法，提取线段轮廓，计算卡片位置，提取卡片图块并校正到指定大小
        Mat card = getCard(grey, threshold, debug, tempPath);

        // 将卡片切图，由起点到中轴线(忽略人像的影响)，计算水平方向投影，从而确定文字所在的行

        // 再次提取轮廓，主要提取文字所在位置的轮廓
        Rect rect = null;

        // 定向识别文字  // 身份证的文字，可以直接按黑色提取
        //        recoChars(new File("D:\\CardDetect\\test\\num.jpg"), rect);
        //        recoChars(new File("D:\\CardDetect\\test\\name.jpg"), rect);
        //        recoChars(new File("D:\\CardDetect\\test\\gender.jpg"), rect);
        //        recoChars(new File("D:\\CardDetect\\test\\address.jpg"), rect);
    }


    public static void main(String[] args) {
        Instant start = Instant.now();
        Mat src = Imgcodecs.imread("D:/CardDetect/4.jpg");
        Boolean debug = true;
        String tempPath = TEMP_PATH + "";

        new IdCardUtil(); // 调用构造方法，加载模型文件
        cardDetect(src, debug, tempPath); // 检测并识别卡片文字

        Instant end = Instant.now();
        System.err.println("总耗时：" + Duration.between(start, end).toMillis());
    }

}
