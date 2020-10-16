package com.yuxue.util;

import java.io.File;
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
            Imgcodecs.imwrite(tempPath + GenerateIdUtil.getId() +"_" +methodName + ".jpg", inMat);
        }
    }


    /**
     * 图像灰度化
     * @param inMat rgbMat/原图
     * @param debug 是否输出结果图片
     * @param tempPath 结果图片输出路径
     * @return greyMat
     */
    public static Mat gray(Mat inMat, Boolean debug, String tempPath) {
        Mat dst = new Mat();
        Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2GRAY);
        // debugImg(false, tempPath, "gray", dst);
        return dst;
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
        // debugImg(debug, tempPath, "gaussianBlur", dst);
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
        // debugImg(debug, tempPath, "blur", dst);
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

        //注意求梯度的时候我们使用的是Scharr算法，sofia算法容易受到图像细节的干扰
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
     * 使用闭操作。对图像进行闭操作以后，可以看到车牌区域被连接成一个矩形的区域
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
        /*if(inMat.width() > 600) {
            offset = new Point(-5, -10); // 偏移量 // 对应sobel的偏移量
        }*/
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
    public static Vector<Mat> screenBlock(Mat src, List<MatOfPoint> contours, Boolean isGreen,Boolean debug, String tempPath){
        Vector<Mat> dst = new Vector<Mat>();

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint m1 = contours.get(i);
            MatOfPoint2f m2 = new MatOfPoint2f(m1.toArray());

            // RotatedRect 该类表示平面上的旋转矩形，有三个属性： 矩形中心点(质心); 边长(长和宽); 旋转角度
            // boundingRect()得到包覆此轮廓的最小正矩形， minAreaRect()得到包覆轮廓的最小斜矩形
            RotatedRect mr = Imgproc.minAreaRect(m2);

            // 以图片左上角为原点，上边为x轴建立坐标系；
            // x轴逆时针旋转，首次平行的边为mr.size.width，x轴跟该条边组成的角度，即angle，  角度取值范围：[-90° ~ 0°]
            double angle = mr.angle;
            if (checkPlateSize(mr)) {  //排除不合法的图块
                if (debug) {
                    Mat result = src.clone();
                    // 外接斜矩形 描绘到原图 
                    Mat points = new Mat();
                    Imgproc.boxPoints(mr, points);
                    if(points.rows() == 4) {
                        Point start = new Point(points.get(0, 0)[0], points.get(0, 1)[0]);
                        Point end = new Point(points.get(1, 0)[0], points.get(1, 1)[0]);
                        Imgproc.line(result, start, end, new Scalar(0, 255, 0, 255));
                        start = new Point(points.get(1, 0)[0], points.get(1, 1)[0]);
                        end = new Point(points.get(2, 0)[0], points.get(2, 1)[0]);
                        Imgproc.line(result, start, end, new Scalar(0, 255, 0, 255));
                        start = new Point(points.get(2, 0)[0], points.get(2, 1)[0]);
                        end = new Point(points.get(3, 0)[0], points.get(3, 1)[0]);
                        Imgproc.line(result, start, end, new Scalar(0, 255, 0, 255));
                        start = new Point(points.get(3, 0)[0], points.get(3, 1)[0]);
                        end = new Point(points.get(0, 0)[0], points.get(0, 1)[0]);
                        Imgproc.line(result, start, end, new Scalar(0, 255, 0, 255));
                    }
                    // 将轮廓描绘到原图   
                    Imgproc.drawContours(result, Lists.newArrayList(m1), -1, new Scalar(0, 0, 255, 255));
                    // 输出带轮廓的原图
                    debugImg(debug, tempPath, "crop", result);
                }

                Size rect_size = new Size((int) mr.size.width, (int) mr.size.height);
                if (mr.size.width < mr.size.height) {
                    angle = 90 + angle; // 处理车牌相对水平位置，旋转角度不超过90°的图片，超过之后，车牌相当于倒置，不予处理
                    rect_size = new Size(rect_size.height, rect_size.width);
                }
                /*System.err.println("外接矩形倾斜角度：" +  mr.angle);
                System.err.println("校正角度：" +  angle);*/

                // 旋转角度，根据需要是否进行角度旋转
                Mat img_rotated = new Mat();
                Mat rotmat = Imgproc.getRotationMatrix2D(mr.center, angle, 1); // 旋转对象；angle>0则 逆时针
                // 如果相机在车牌正前方，拍摄角度较小，不管相机是否保持水平，使用仿射变换，减少照片倾斜影响即可
                // 如果相机在车牌的左前、右前、上方等，拍摄角度较大时，则需要考虑使用投影变换
                Imgproc.warpAffine(src, img_rotated, rotmat, src.size()); // 仿射变换  对原图进行旋转校正 // 处理倾斜的图片
                debugImg(debug, tempPath, "img_rotated", img_rotated);

                // 仿射变换  对原图进行错切校正
                // 轮廓的提取，直接影响校正的效果
                Mat shear = img_rotated.clone();
                rect_size = shearCorrection(m2, mr, img_rotated, shear, rect_size, debug, tempPath);

                // 切图   按给定的尺寸、给定的中心点
                Mat img_crop = new Mat();

                if(isGreen) {
                    // 如果是新能源牌照，需要向上扩展一定的尺寸
                    Size s = new Size(rect_size.width, rect_size.height + (rect_size.height/8));
                    Point c = new Point(mr.center.x, mr.center.y - (rect_size.height/16) -8);   // 偏移量修正
                    Imgproc.getRectSubPix(shear, s, c, img_crop);
                } else {
                    Point c = new Point(mr.center.x, mr.center.y -4);   // 偏移量修正
                    Imgproc.getRectSubPix(img_rotated, rect_size, c, img_crop);
                }

                // 处理切图，调整为指定大小
                Mat resized = new Mat(Constant.DEFAULT_HEIGHT, Constant.DEFAULT_WIDTH, TYPE);
                Imgproc.resize(img_crop, resized, resized.size(), 0, 0, Imgproc.INTER_CUBIC); // INTER_AREA 缩小图像的时候使用 ; INTER_CUBIC 放大图像的时候使用
                debugImg(true, tempPath, "crop_resize", resized);
                dst.add(resized);
            }
        }
        return  dst;
    }


    /**
     * 图块错切校正
     * 根据轮廓、以及最小斜矩形矩形错切校正，用于处理变形(不是倾斜)的车牌图片  即【平行四边形】的车牌校正为【长方形】
     * 该算法，容易受到到轮廓的影响，要求轮廓定位得比较精确 
     * 其他方案: 
     *  1、在处理字符的时候，进行错切校正，根据字符的外接矩形倾斜角度来校正字符即可
     *  2、在训练字符识别模型的时候，加入错切的样本数据
     * @param m2 轮廓
     * @param mr 包覆轮廓的最小斜矩形
     * @param inMat 原图
     * @param shear 错切校正后的图
     * @param rect_size 斜矩形的size
     * @param debug
     * @param tempPath
     * @return
     */
    private static Size shearCorrection(MatOfPoint2f m2, RotatedRect mr, Mat inMat, Mat shear, Size rect_size, Boolean debug, String tempPath){
        Mat vertex = new Mat(); 
        Imgproc.boxPoints(mr, vertex);  // 最小外接矩形，四个顶点 Mat(4, 2)
        // 提取短边的两个顶点， 命名为上、下顶点
        Point p0 = new Point(vertex.get(0, 0)[0], vertex.get(0, 1)[0]);
        Point p1 = new Point(vertex.get(1, 0)[0], vertex.get(1, 1)[0]);
        Point p2 = new Point(vertex.get(2, 0)[0], vertex.get(2, 1)[0]);
        Point p3 = new Point(vertex.get(3, 0)[0], vertex.get(3, 1)[0]);
        Point[] shortLine0 = {p0, p1}; // 短边
        Point[] shortLine1 = {p2, p3}; // 短边
        Point[] longLine0 = {p0, p3}; // 长边
        Point[] longLine1 = {p2, p1}; // 长边
        if(getDistance(p0, p1) > getDistance(p0, p3)) {
            shortLine0[1] = p3;
            shortLine1[1] = p1;
            longLine0[1] = p1;
            longLine1[1] = p3;
        }

        Point[] leftShortLine = shortLine0;
        if(shortLine0[0].x + shortLine0[1].x > shortLine1[0].x + shortLine1[1].x ) { // 只要有错切，就一定不相等
            leftShortLine = shortLine1;
        }

        double height = mr.size.height;
        double width = mr.size.width;
        if(width < height) {
            height = mr.size.width;
            width = mr.size.height;
        }
        // 最小外接矩形，两条长的边，一定是跟轮廓的长的边保持平行的
        // 根据轮廓计算校正像素值； 错切像素取值范围：[5,30]以内，否则不予处理，防止动作较大，影响结果
        // 计算，轮廓里面，离短边最近的点，获取其距离短边的距离
        List<Point> points = m2.toList();
        if(null == points || points.size() <= 0) {
            return rect_size;
        }

        List<Point> result = Lists.newArrayList(); // 按坐标有序
        double distanceSum = 0;
        // 遍历，获取离短边最近轮廓短边的点集合; 计算错切值
        for (Point p : points) {
            // 排除离两条长边较近的点
            if(getDistance(p, longLine0[0], longLine0[1]) <= height/5) {
                continue;
            }
            if(getDistance(p, longLine1[0], longLine1[1]) <= height/5) {
                continue;
            }
            // 提取剩下点中，离左短边较近的点
            double distance = getDistance(p, leftShortLine[0], leftShortLine[1]);
            if(distance <= width/4) {
                result.add(p);
                distanceSum += distance;
            }
        }
        // 计算到的错切值，大于0，则上边线向右，下边线向左拉伸； 小于0，则上边线向左，下边线向右拉伸
        double shearPX = 2 * distanceSum / result.size();
        if( Constant.DEFAULT_MIN_SHEAR_PX > shearPX || shearPX > Constant.DEFAULT_MAX_SHEAR_PX) {
            return rect_size;
        }
        // 距离跟坐标排序，判断需要错切的方向
        // 任取两个点得到一条直线，得到跟短边的交点，
        // 交点距up点较近，则向右拉伸，交点距离down点较近，则向左拉伸
        Point up = leftShortLine[0];
        Point down = leftShortLine[1];
        if(up.y > down.y) {
            up = leftShortLine[1];
            down = leftShortLine[0];
        }

        Integer c1 = 0, c2 = 0; // c1>c2,则向右拉伸
        for (int i = 0; i < result.size() / 2; i++) {
            Point s = result.get(i);
            for (int j = result.size() / 2; j < result.size(); j++) {
                Point e = result.get(j);
                if(getDistance(up, s, e) < getDistance(down, s, e) ) {
                    c1++;
                } else {
                    c2++;
                }
            }
        }
        // 错切校正之后，要减掉校正的像素值；校正前轮廓外接矩形较大，校正后，需要调整矩形的width
        if(rect_size.width - shearPX > 0) {
            rect_size = new Size(rect_size.width - shearPX, rect_size.height);
        }
        if(c1<c2) {
            shearPX = -shearPX;
        }
        // System.err.println("错切方向： " + shearPX);
        // 计算错切比例
        double top = shearPX / height *  down.y; 
        double bottom = shearPX / height * (inMat.height() - up.y); 

        // 提取图片左上、左下、右上 三个顶点，根据角度，计算偏移量
        MatOfPoint2f srcPoints = new MatOfPoint2f();
        srcPoints.fromArray(new Point(0, 0), new Point(0, inMat.rows()), new Point(inMat.cols(), 0));
        MatOfPoint2f dstPoints = new MatOfPoint2f();
        dstPoints.fromArray(new Point(0 + top, 0), new Point(0 - bottom, inMat.rows()), new Point(inMat.cols() + top, 0));

        Mat m3 = Imgproc.getAffineTransform(srcPoints, dstPoints);
        Imgproc.warpAffine(inMat, shear, m3, inMat.size()); // 对整张图进行错切校正

        // 投影变换举例; 对于车牌的处理效果来说，跟三点法差不多，但是效率慢
        /*MatOfPoint2f srcPoints = new MatOfPoint2f();
        srcPoints.fromArray(new Point(0, 0), new Point(0, inMat.rows()), new Point(inMat.cols(), 0), new Point(inMat.cols(), inMat.rows()));
        MatOfPoint2f dstPoints = new MatOfPoint2f();
        dstPoints.fromArray(new Point(0 + 80, 0), new Point(0 - 80, inMat.rows()), new Point(inMat.cols() + 80, 0) , new Point(inMat.cols() - 80, inMat.rows()));
        Mat m3 = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
        Imgproc.warpPerspective(inMat, shear, m3, inMat.size());*/

        debugImg(debug, tempPath, "shearCorrection", shear);
        return rect_size;
    }


    /**
     * 计算两个点之间的距离
     * @param p1
     * @param p2
     * @return
     */
    public static double getDistance(Point p1, Point p2) {
        double distance = 0;
        distance = Math.pow((p1.x - p2.x), 2) + Math.pow((p1.y - p2.y), 2);
        distance = Math.sqrt(distance);
        return distance;
    }


    /**
     * 计算点到AB点的距离
     * 即，计算点到线的垂直距离
     * @param p
     * @param a
     * @param b
     * @return
     */
    public static double getDistance(Point p, Point a, Point b) {
        double distance = 0, A = 0, B = 0, C = 0;
        A = a.y - b.y;
        B = b.x - a.x;
        C = a.x * b.y - a.y * b.x;
        // 代入点到直线距离公式
        distance = (Math.abs(A * p.x + B * p.y + C)) / (Math.sqrt(A * A + B * B));
        return distance;
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
        int min = Constant.DEFAULT_MIN_SIZE;
        int max = Constant.DEFAULT_MAX_SIZE;
        // 计算切图面积
        int area = (int) (mr.size.height * mr.size.width);
        // 计算切图宽高比
        double r = mr.size.width / mr.size.height;
        if (r < 1) {  // 特殊情况下，获取到的width  height 值是相反的
            r = mr.size.height / mr.size.width;
        }
        return min <= area && area <= max && Constant.DEFAULT_MIN_RATIO <= r && r <= Constant.DEFAULT_MAX_RATIO;
    }


    /**
     * 进行膨胀操作
     * 也可以理解为字体加粗操作
     * @param inMat
     * @return
     */
    public static Mat dilate(Mat inMat, Boolean debug, String tempPath, int row, int col, Boolean correct) {
        Mat result = inMat.clone();
        // 返回指定形状和尺寸的结构元素  矩形：MORPH_RECT;交叉形：MORPH_CROSS; 椭圆形：MORPH_ELLIPSE
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(row, col));
        Imgproc.dilate(inMat, result, element);

        // 先腐蚀 后扩张，会存在一定的偏移； 这里校正偏移量
        if(correct) {
            Mat transformMat = Mat.eye(2, 3, CvType.CV_32F);
            transformMat.put(0, 2, -col/2);
            transformMat.put(1, 2, -row/2);
            Imgproc.warpAffine(result, result, transformMat, inMat.size());
        }
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
     * HSV色彩空间过滤
     * @param inMat rgb图像
     * @param debug
     * @param tempPath
     * @param hsv 变长参数，依次为： minH,maxH,minS,maxS,minV,maxV
     * @return 返回过滤后的hsvMat; 不满足range的像素点，替换为黑色
     */
    public static Mat hsvFilter(Mat inMat, Boolean debug, String tempPath, Integer...range) {
        Mat hsvMat = new Mat();    // 转换为hsv图像
        Imgproc.cvtColor(inMat, hsvMat, Imgproc.COLOR_BGR2HSV);
        Mat dst = hsvMat.clone();
        // 从数据库中读取配置参数
        for (int i = 0; i < hsvMat.rows(); i++) {
            for (int j = 0; j < hsvMat.cols(); j++) {
                double[] hsv = hsvMat.get(i, j);
                Integer h = (int) hsv[0];
                Integer s = (int) hsv[1];
                Integer v = (int) hsv[2];
                Integer c = 0;
                if (range.length >= 1 && range[0] <= h) {
                    c++;
                }
                if (range.length >= 2 && h <= range[1]) {
                    c++;
                }
                if (range.length >= 3 && range[2] <= s) {
                    c++;
                }
                if (range.length >= 4 && s <= range[3]) {
                    c++;
                }
                if (range.length >= 5 && range[4] <= v) {
                    c++;
                }
                if (range.length >= 6 && v <= range[5]) {
                    c++;
                }
                if (c == range.length) {   // 所有条件都满足，不处理
                    continue;
                } else {
                    hsv[0] = 0.0;
                    hsv[1] = 0.0;
                    hsv[2] = 0.0;   // 黑色
                    dst.put(i, j, hsv);
                }
            }
        }
        debugImg(debug, tempPath, "hsvFilter", dst);
        return dst;
    }


    /**
     * HSV色彩空间过滤
     * @param inMat rgb图像
     * @param debug
     * @param tempPath
     * @param hsv 变长参数，依次为： minH,maxH,minS,maxS,minV,maxV
     * @return 返回二值图像
     */
    public static Mat hsvThreshold(Mat inMat, Boolean debug, String tempPath, Integer...range) {
        Mat hsvMat = new Mat();    // 转换为hsv图像
        Imgproc.cvtColor(inMat, hsvMat, Imgproc.COLOR_BGR2HSV);
        Mat threshold = new Mat(hsvMat.size(), hsvMat.type());
        for (int i = 0; i < hsvMat.rows(); i++) {
            for (int j = 0; j < hsvMat.cols(); j++) {
                double[] hsv = hsvMat.get(i, j);
                Integer h = (int) hsv[0];
                Integer s = (int) hsv[1];
                Integer v = (int) hsv[2];
                Integer c = 0;
                if (range.length >= 1 && range[0] <= h) {
                    c++;
                }
                if (range.length >= 2 && h <= range[1]) {
                    c++;
                }
                if (range.length >= 3 && range[2] <= s) {
                    c++;
                }
                if (range.length >= 4 && s <= range[3]) {
                    c++;
                }
                if (range.length >= 5 && range[4] <= v) {
                    c++;
                }
                if (range.length >= 6 && v <= range[5]) {
                    c++;
                }
                if (c == range.length) {   // 所有条件都满足，不处理
                    hsv[0] = 255.0;
                    hsv[1] = 255.0;
                    hsv[2] = 255.0; // 白色
                } else {
                    hsv[0] = 0.0;
                    hsv[1] = 0.0;
                    hsv[2] = 0.0;   // 黑色 二值算法
                }
                threshold.put(i, j, hsv);
            }
        }
        debugImg(debug, tempPath, "hsvThreshold", threshold);
        return threshold;
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

        // INTER_AREA 缩小图像的时候使用 // INTER_CUBIC 放大图像的时候使用
        double fx = (double)resized.cols()/inMat.cols(); // 水平缩放比例，输入为0时，则默认当前计算方式
        double fy = (double)resized.rows()/inMat.rows(); // 垂直缩放比例，输入为0时，则默认当前计算方式
        Imgproc.resize(inMat, resized, resized.size(), fx, fy, Imgproc.INTER_LINEAR);
        // debugImg(debug, tempPath, "resizeMat", resized); // 不再生成debug图片
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


    public static void main(String[] args) {
        Mat shear = new Mat();  // 校正后的图片

        String tempPath = Constant.DEFAULT_TEMP_DIR + "test/";
        String filename = tempPath + "15.jpg";
        File f = new File(filename);
        if(!f.exists()) {
            File f1 = new File(filename.replace("jpg", "png"));
            File f2 = new File(filename.replace("png", "bmp"));
            filename = f1.exists() ? f1.getPath() : f2.getPath();
        }

        Mat inMat = Imgcodecs.imread(filename);
        // 提取图片左上、左下、右上 三个顶点，根据角度，计算偏移量
        MatOfPoint2f srcPoints = new MatOfPoint2f();
        srcPoints.fromArray(new Point(0, 0), new Point(0, inMat.rows()), new Point(inMat.cols(), 0));
        MatOfPoint2f dstPoints = new MatOfPoint2f();
        dstPoints.fromArray(new Point(0 - 180, 0), new Point(0 + 180, inMat.rows()), new Point(inMat.cols() - 180, 0)); // 上边线向左，下边线向右拉伸

        // 对整张图进行错切校正
        Mat m3 = Imgproc.getAffineTransform(srcPoints, dstPoints);
        Imgproc.warpAffine(inMat, shear, m3, inMat.size());
        debugImg(true, tempPath, "shearCorrection", shear);
    }


}
