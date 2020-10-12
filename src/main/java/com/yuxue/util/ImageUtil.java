package com.yuxue.util;

import java.util.List;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.google.common.collect.Lists;
import com.yuxue.constant.Constant;

/**
 * 图片处理工具类
 * 将原图，经过算法处理，得到车牌的图块
 * @author yuxue
 * @date 2020-05-18 12:07
 */
public class ImageUtil {


    static {
        // 加载本地安装的opencv库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    /**
     * 生成文件名称
     * @return
     */
    private static Integer id = 1000;
    public static synchronized Integer getId() {
        if(id == Integer.MAX_VALUE) {
            id = 1000;
        }
        return id++;
    }


    /***
     * 保存算法过程每个步骤处理结果，输出结果jpg图像
     * @param debug 缓存目录
     * @param tempPath
     * @param methodName
     * @param inMat
     */
    public static void debugImg(Boolean debug, String tempPath, String methodName, Mat inMat) {
        if (debug) {
            // 通过getId生成文件名称，使得每个步骤生成的图片能够按照执行时间进行排序
            Imgcodecs.imwrite(tempPath + getId() +"_" +methodName + ".jpg", inMat);
        }
    }


    /**
     * 高斯滤波，用于 抑制噪声，平滑图像， 防止把噪点也检测为边缘
     * 高斯滤波器相比于均值滤波器对图像个模糊程度较小
     * https://blog.csdn.net/qinchao315/article/details/81269328
     * https://blog.csdn.net/qq_35294564/article/details/81142524
     * @param inMat 原图
     * @param debug 是否输出结果图片
     * @param tempPath 结果图片输出路径
     * @return
     */
    public static final int GS_BLUR_KERNEL = 3;  // 滤波内核大小必须是 正奇数
    public static Mat gaussianBlur(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Size ksize = new Size(GS_BLUR_KERNEL, GS_BLUR_KERNEL);
        Imgproc.GaussianBlur(inMat, dst, ksize, 0, 0, Core.BORDER_DEFAULT);
        debugImg(debug, tempPath, "gaussianBlur", dst);
        return dst;
    }


    /**
     * 均值滤波
     * @param inMat
     * @param debug 是否输出结果图片
     * @param tempPath 结果图片输出路径
     * @return
     */
    public static final int BLUR_KERNEL = 5;  // 滤波内核大小必须是 正奇数
    public static Mat blur(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Point anchor = new Point(-1,-1);
        Size ksize = new Size(BLUR_KERNEL, BLUR_KERNEL);
        Imgproc.blur(inMat, dst, ksize, anchor, Core.BORDER_DEFAULT);
        debugImg(debug, tempPath, "blur", dst);
        return dst;
    }


    /**
     * 图像灰度化
     * @param inMat 高斯滤波后的图
     * @param debug 是否输出结果图片
     * @param tempPath 结果图片输出路径
     * @return
     */
    public static Mat gray(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2GRAY);
        debugImg(false, tempPath, "gray", dst);
        return dst;
    }


    /**
     * 对图像进行Sobel 运算，得到图像的一阶水平方向导数
     * 边缘检测算子，是一阶的梯度算法
     * 所谓梯度运算就是对图像中的像素点进行就导数运算，从而得到相邻两个像素点的差异值
     * 对噪声具有平滑作用，提供较为精确的边缘方向信息，边缘定位精度不够高。当对精度要求不是很高时，是一种较为常用的边缘检测方法
     * @param inMat 灰度图
     * @param debug
     * @param tempPath
     * @return
     */
    public static final int SOBEL_SCALE = 1;
    public static final int SOBEL_DELTA = 0;
    public static final int SOBEL_X_WEIGHT = 1;
    public static final int SOBEL_Y_WEIGHT = 0;
    public static final int SOBEL_KERNEL = 3;// 内核大小必须为奇数且不大于31
    public static final double alpha = 1.5; // 乘数因子
    public static final double beta = 10.0; // 偏移量
    public static Mat sobel(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();

        // Sobel滤波 计算水平方向灰度梯度的绝对值
        Imgproc.Sobel(inMat, grad_x, CvType.CV_8U, 1, 0, SOBEL_KERNEL, SOBEL_SCALE, SOBEL_DELTA, Core.BORDER_DEFAULT); 
        Core.convertScaleAbs(grad_x, abs_grad_x, alpha, beta);   // 增强对比度

        // Sobel滤波 计算垂直方向灰度梯度的绝对值
        Imgproc.Sobel(inMat, grad_y, CvType.CV_8U, 0, 1, SOBEL_KERNEL, SOBEL_SCALE, SOBEL_DELTA, Core.BORDER_DEFAULT);
        Core.convertScaleAbs(grad_y, abs_grad_y, alpha, beta);
        grad_x.release();
        grad_y.release();

        // 计算结果梯度
        Core.addWeighted(abs_grad_x, SOBEL_X_WEIGHT, abs_grad_y, SOBEL_Y_WEIGHT, 0, dst);
        abs_grad_x.release();
        abs_grad_y.release();

        debugImg(debug, tempPath, "sobel", dst);
        return dst;
    }


    /**
     * 对图像进行scharr 运算，得到图像的一阶水平方向导数
     * 增强对比度，边缘检测
     * 所谓梯度运算就是对图像中的像素点进行就导数运算，从而得到相邻两个像素点的差异值
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat scharr(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();

        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();

        //注意求梯度的时候我们使用的是Scharr算法，sofia算法容易收到图像细节的干扰
        Imgproc.Scharr(inMat, grad_x, CvType.CV_32F, 1, 0);
        Imgproc.Scharr(inMat, grad_y, CvType.CV_32F, 0, 1);
        //openCV中有32位浮点数的CvType用于保存可能是负值的像素数据值
        Core.convertScaleAbs(grad_x, abs_grad_x);
        Core.convertScaleAbs(grad_y, abs_grad_y);
        //openCV中使用release()释放Mat类图像，使用recycle()释放BitMap类图像
        grad_x.release();
        grad_y.release();

        Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
        abs_grad_x.release();
        abs_grad_y.release();

        debugImg(debug, tempPath, "scharr", dst);
        return dst;
    }


    /**
     * 对图像进行二值化。将灰度图像（每个像素点有256个取值可能， 0代表黑色，255代表白色）  
     * 转化为二值图像（每个像素点仅有1和0两个取值可能）
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat threshold(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Imgproc.threshold(inMat, dst, 100, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);
        debugImg(debug, tempPath, "threshold", dst);
        inMat.release();
        return dst;
    }


    /**
     * 使用闭操作。对图像进行闭操作以后，可以看到车牌区域被连接成一个矩形装的区域
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static final int DEFAULT_MORPH_SIZE_WIDTH = 10;
    public static final int DEFAULT_MORPH_SIZE_HEIGHT = 10; // 大于1
    public static Mat morphology(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat(inMat.size(), CvType.CV_8UC1);
        Size size = new Size(DEFAULT_MORPH_SIZE_WIDTH, DEFAULT_MORPH_SIZE_HEIGHT);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size);
        Imgproc.morphologyEx(inMat, dst, Imgproc.MORPH_CLOSE, element);
        debugImg(debug, tempPath, "morphology", dst);
        return dst;
    }


    /**
     * 提取外部轮廓
     * 这个算法会把全图的轮廓都计算出来，因此要进行筛选。
     * @param src 原图
     * @param inMat morphology Mat
     * @param debug
     * @param tempPath
     * @return
     */
    public static List<MatOfPoint> contours(Mat src, Mat inMat, Boolean debug, String tempPath) {
        List<MatOfPoint> contours = Lists.newArrayList();
        Mat hierarchy = new Mat();
        // RETR_EXTERNAL只检测最外围轮廓， // RETR_LIST   检测所有的轮廓
        // CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
        Point offset = new Point(0, 0); // 偏移量
        if(inMat.width() > 600) {
            offset = new Point(0, -10); // 偏移量 // 对应sobel的偏移量
        }
        Imgproc.findContours(inMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE, offset);
        if (debug) {
            Mat result = new Mat();
            src.copyTo(result); //  复制一张图，不在原图上进行操作，防止后续需要使用原图
            // 将轮廓用红色描绘到原图
            Imgproc.drawContours(result, contours, -1, new Scalar(0, 0, 255, 255));
            // 输出带轮廓的原图
            debugImg(debug, tempPath, "contours", result);
        }
        return contours;
    }


    /**
     * 根据轮廓， 筛选出可能是车牌的图块
     * @param src
     * @param matVector
     * @param debug
     * @param tempPath
     * @return
     */
    public static final int DEFAULT_ANGLE = 90; // 角度判断所用常量
    public static final int TYPE = CvType.CV_8UC3;
    public static Vector<Mat> screenBlock(Mat src, List<MatOfPoint> contours, Boolean debug, String tempPath){
        Vector<Mat> dst = new Vector<Mat>();
        List<MatOfPoint> mv = Lists.newArrayList(); // 用于在原图上描绘筛选后的结果

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint m1 = contours.get(i);
            MatOfPoint2f m2 = new MatOfPoint2f();
            m1.convertTo(m2, CvType.CV_32F);
            
            // 多边形逼近 -- 没什么卵用
            /*double epsilon = 0.001 * Imgproc.arcLength(m2, true);
            Imgproc.approxPolyDP(m2, m2, epsilon, true);*/
            
            // RotatedRect 该类表示平面上的旋转矩形，有三个属性： 矩形中心点(质心); 边长(长和宽); 旋转角度
            // boundingRect()得到包覆此轮廓的最小正矩形， minAreaRect()得到包覆轮廓的最小斜矩形
            RotatedRect mr = Imgproc.minAreaRect(m2);
            
            // 以图片左上角为原点，上边为x轴建立坐标系；
            // x轴逆时针旋转，首次平行的边为mr.size.width，x轴跟该条边组成的角度，即angle，  角度取值范围：[-90° ~ 0°]
            double angle = mr.angle;
            if (checkPlateSize(mr) && Math.abs(mr.angle) <= DEFAULT_ANGLE) {  //排除不合法的图块
                mv.add(contours.get(i));
                Size rect_size = new Size((int) mr.size.width, (int) mr.size.height);
                if (mr.size.width < mr.size.height) {
                    angle = 90 + angle; // 处理车牌相对水平位置，旋转角度不超过90°的图片，超过之后，车牌相当于倒置，不予处理
                    rect_size = new Size(rect_size.height, rect_size.width);
                }
                // 旋转角度，根据需要是否进行角度旋转
                Mat img_rotated = new Mat();
                Mat rotmat = Imgproc.getRotationMatrix2D(mr.center, angle, 1); // 旋转对象；angle>0则 逆时针
                // 如果相机在车牌正前方，拍摄角度较小，不管相机是否保持水平，使用仿射变换，减少照片倾斜影响即可
                // 如果相机在车牌的左前、右前、上方等，拍摄角度较大时，则需要考虑使用投影变换
                Imgproc.warpAffine(src, img_rotated, rotmat, src.size()); // 仿射变换  对原图进行旋转 

                // 仿射变换  对原图进行错切
                // 获取轮廓四个顶点，来判断是否需要进行错切， 取三个点计算即可 --未完成yuxue
                
                
                
                // 切图
                Mat img_crop = new Mat();
                Imgproc.getRectSubPix(img_rotated, rect_size, mr.center, img_crop);
                debugImg(debug, tempPath, "crop", img_crop);

                // 处理切图，调整为指定大小
                Mat resized = new Mat(Constant.DEFAULT_HEIGHT, Constant.DEFAULT_WIDTH, TYPE);
                Imgproc.resize(img_crop, resized, resized.size(), 0, 0, Imgproc.INTER_CUBIC); // INTER_AREA 缩小图像的时候使用 ; INTER_CUBIC 放大图像的时候使用
                // Imgproc.getPerspectiveTransform(img_crop, resized); // 投影变换
                debugImg(debug, tempPath, "crop_resize", resized);
                dst.add(resized);
            }
        }
        if (debug) {
            Mat result = new Mat();
            src.copyTo(result);
            // 将轮廓描绘到原图
            Imgproc.drawContours(result, mv, -1, new Scalar(0, 0, 255, 255));
            // 输出带轮廓的原图
            debugImg(debug, tempPath, "screenblock", result);
        }
        return  dst;
    }

    
    /**
     * 图块错切校正
     * 
     * @return
     */
    private static Mat shearCorrection(MatOfPoint2f m2, Mat inMat){
        Mat shear = new Mat();  // 校正后的图片
        
        
        return shear;
    }
    /**
     * 对minAreaRect获得的最小外接矩形
     * 判断面积以及宽高比是否在制定的范围内
     * 黄牌、蓝牌、绿牌
     * 国内车牌大小: 440mm*140mm，宽高比 3.142857
     * @param mr
     * @return
     */
    private static boolean checkPlateSize(RotatedRect mr) {
        // 切图面积取值范围
        int min = 44 * 14 * 1;
        int max = 44 * 14 * 40;
        // 计算切图面积
        int area = (int) (mr.size.height * mr.size.width);
        // 计算切图宽高比
        double r = mr.size.width / mr.size.height;
        if (r < 1) {  // 特殊情况下，获取到的width  height 值是相反的
            r = mr.size.height / mr.size.width;
        }
        return min <= area && area <= max && 2 <= r && r <= 10;
    }


    /**
     * 进行膨胀操作
     * 也可以理解为字体加粗操作
     * @param inMat
     * @return
     */
    public static Mat dilate(Mat inMat, Boolean debug, String tempPath, int row, int col) {
        Mat result = inMat.clone();
        // 返回指定形状和尺寸的结构元素  矩形：MORPH_RECT;交叉形：MORPH_CROSS; 椭圆形：MORPH_ELLIPSE
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(row, col));
        Imgproc.dilate(inMat, result, element);
        debugImg(debug, tempPath, "dilate", result);
        return result;
    }

    /**
     * 进行腐蚀操作
     * @param inMat
     * @return
     */
    public static Mat erode(Mat inMat, Boolean debug, String tempPath, int row, int col) {
        Mat result = inMat.clone();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(row, col));
        Imgproc.erode(inMat, result, element);
        debugImg(debug, tempPath, "erode", result);
        return result;
    }



    /**
     * 直方图均衡化   用于提高图像的质量
     * 直方图均衡化是一种常见的增强图像对比度的方法，使用该方法可以增强局部图像的对比度，尤其在数据较为相似的图像中作用更加明显
     * 直方图是图像中像素强度分布的图形表达方式.
     * 它统计了每一个强度值所具有的像素个数
     * https://blog.csdn.net/zqx951102/article/details/84201936
     * @param inMat
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat equalizeHist(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        // 灰度图均衡化
        // Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2GREY);
        // Imgproc.equalizeHist(inMat, dst); 

        // 转到HSV空间进行均衡化     (H色相     S饱和度     V亮度)
        Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2HSV);
        List<Mat> hsvSplit = Lists.newArrayList();
        Core.split(dst, hsvSplit); // 通道分离  三个通道是反过来的  0:V 1:S 2:H
        Imgproc.equalizeHist(hsvSplit.get(2), hsvSplit.get(2)); // 对H(亮度)空间进行均衡化
        Core.merge(hsvSplit, dst);
        debugImg(debug, tempPath, "equalizeHist", dst);
        return dst;
    }



    /**
     * 颜色范围提取
     * @param grey
     * @param tempPath
     * @param debug
     * @return
     */
    public static Mat hsvFilter(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = inMat.clone();
        Mat hsvMat = new Mat();    // 转换为hsv图像
        Imgproc.cvtColor(inMat, hsvMat, Imgproc.COLOR_BGR2HSV);
        // 从数据库中读取配置参数
        // 蓝牌
        /*Scalar lowerB = new Scalar(new double[] { 110, 230, 183 });
        Scalar upperB = new Scalar(new double[] { 120, 250, 229 });
        // 绿牌
        Scalar lowerG = new Scalar(new double[] { 75, 166, 194 });
        Scalar upperG = new Scalar(new double[] { 80, 38, 183 });
        // 黄牌
        Scalar lowerY = new Scalar(new double[] { 20, 188, 253 });
        Scalar upperY = new Scalar(new double[] { 30, 188, 253 });*/

        for (int i = 0; i < hsvMat.rows(); i++) {
            for (int j = 0; j < hsvMat.cols(); j++) {
                double[] hsv = hsvMat.get(i, j);
                Integer h = (int)hsv[0];
                Integer s = (int)hsv[1];
                Integer v = (int)hsv[2];
                if (105 <= h && h <= 125 && 100 <= s && s <= 255 && 50 <= v && v <= 200) {
                    continue;
                } else {
                    hsv[0] = 255.0;
                    hsv[1] = 255.0;
                    hsv[2] = 255.0;
                    dst.put(i, j, hsv);
                }
            }
        }
        debugImg(debug, tempPath, "hsvFilter", dst);
        return dst;
    }


    /**
     * 锁定横纵比，调整图片大小(缩小)
     * 防止图片像素太大，后续的计算太费时
     * 但是这样处理之后，图片可能会失真，影响车牌文字识别效果
     * 可以考虑，定位出车牌位置之后，计算出原图的车牌位置，从原图中区图块进行车牌文字识别
     * @param inMat
     * @param maxRows
     * @return
     */
    public static Mat resizeMat(Mat inMat, Integer maxCols, Boolean debug, String tempPath) {
        if(null == maxCols || maxCols <= 0) {
            maxCols = 600;
        }
        if(maxCols >= inMat.cols()) {   // 图片尺寸小于指定大小，则不处理
            return inMat;
        }
        float r = inMat.rows() * 1.0f / inMat.cols(); 
        Integer rows = Math.round(maxCols * r);
        Mat resized = new Mat(rows, maxCols, inMat.type());

        /**
         * INTER_AREA 缩小图像的时候使用
         * INTER_CUBIC 放大图像的时候使用
         */
        double fx = (double)resized.cols()/inMat.cols(); // 水平缩放比例，输入为0时，则默认当前计算方式
        double fy = (double)resized.rows()/inMat.rows(); // 垂直缩放比例，输入为0时，则默认当前计算方式
        Imgproc.resize(inMat, resized, resized.size(), fx, fy, Imgproc.INTER_LINEAR);
        debugImg(true, tempPath, "resizeMat", resized);
        return resized;
    }

    
    /**
     * 还原图片的尺寸(放大)
     * 放大二值图像到原始图片的尺寸，然后提取轮廓，再从原图裁剪图块
     * 防止直接在缩放后的图片上提取图块，因图片变形导致图块识别结果异常
     * @param inMat
     * @param size
     * @param debug
     * @param tempPath
     * @return
     */
    public static Mat restoreSize(Mat inMat, Size size, Boolean debug, String tempPath) {
        if(inMat.width() >= size.width) {
            return inMat;
        }
        Mat restore = new Mat();
        Imgproc.resize(inMat, restore, size, 0, 0, Imgproc.INTER_CUBIC);
        debugImg(debug, tempPath, "restoreSize", restore);
        return restore;
    }
    
    
    


}
