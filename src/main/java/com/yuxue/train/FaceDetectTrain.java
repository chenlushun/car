package com.yuxue.train;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import org.opencv.core.CvType;
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

import com.yuxue.constant.Constant;
import com.yuxue.util.ImageUtil;

/**
 * 基于org.opencv包实现的训练
 * 
 * 人脸识别训练、检测
 * 
 * @author yuxue
 * @date 2020-09-15 12:32
 */
public class FaceDetectTrain {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/FaceDetect/train/";

    // 训练模型文件保存位置
    private static final String MODEL_PATH = DEFAULT_PATH + "haarcascade_frontalface_default.xml";


    public static void detectFace(Mat inMat) {
        Boolean debug = false;
        Mat grey = ImageUtil.gaussianBlur(inMat, debug, DEFAULT_PATH);

        CascadeClassifier faceDetector = new CascadeClassifier(MODEL_PATH);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(grey, faceDetections);

        System.out.println(String.format("识别出 %s 张人脸", faceDetections.toArray().length));
        
        // 在识别到的人脸部位，描框
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(inMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
            Imgcodecs.imwrite(DEFAULT_PATH+ "face.jpg", inMat);
        }
        
        return;
    }

    
    
    public static void main(String[] args) {
        // Mat inMat = Imgcodecs.imread(DEFAULT_PATH + "AverageMaleFace.jpg");
        Mat inMat = Imgcodecs.imread(DEFAULT_PATH + "huge.png");
        // Mat inMat = Imgcodecs.imread(DEFAULT_PATH + "car.jpg");
        FaceDetectTrain.detectFace(inMat);

    }

}
