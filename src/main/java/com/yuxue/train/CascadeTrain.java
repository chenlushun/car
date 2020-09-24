package com.yuxue.train;

import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import com.yuxue.constant.Constant;
import com.yuxue.util.FileUtil;
import com.yuxue.util.ImageUtil;

/**
 * 人脸识别训练、检测 (级联分类器)  未完成
 * 
 * 训练自己的级联分类器 -- 调用opencv的exe应用程序
 * https://blog.csdn.net/dbzzcz/article/details/105517946
 * 获取opencv_traincascade.exe应用程序，官方下载3.4.1版本的安装包，解压出来即可找到
 * 或者加入 1054836232 Q群,在群共享文件中获取应用程序安装包
 * 
 *  官方教程地址：
 *  https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html
 * @author yuxue
 * @date 2020-09-15 12:32
 */
public class CascadeTrain {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // 默认的训练操作的根目录
    private static final String DEFAULT_PATH = "D:/FaceDetect/train/";
    private static final String MODEL_PATH = DEFAULT_PATH + "haarcascade_frontalface_my.xml";
    
    /**
     * 生成文件名称
     * @return
     */
    public static synchronized Long getId() {
        return System.currentTimeMillis();
    }


    /**
     * 读取正负样本文件； 开始训练
     * 训练出来的模型文件，用于识别图片中是否包含人脸
     */
    public static void train() {
        // 正负样本文件路径
        String negative = "D:\\FaceDetect\\samples\\negative\\";
        String positive = "D:\\FaceDetect\\samples\\positive\\";

        Mat samples = new Mat(); // 使用push_back，行数列数不能赋初始值
        Vector<Integer> labels = new Vector<Integer>();

        // 加载正负样本文件
        loadSamples(samples, labels, negative, positive);
        
        // 开始训练 // 使用opencv提供的训练器，命令方式启动
        String _cascadeDirName = "";
        String _posFilename = ""; 
        String _negFilename = "";
        int _numPos = 2000;
        int _numNeg = 1000;
        int _precalcValBufSize = 1024; 
        int _precalcIdxBufSize = 1024;
        int _numStages = 20;

        boolean baseFormatSave = false;
        double acceptanceRatioBreakValue = -1.0;
        
        /*CvCascadeParams _cascadeParams = new CvCascadeParams();
        CvFeatureParams _featureParams = new CvFeatureParams();
        CvCascadeBoostParams _stageParams = new CvCascadeBoostParams();

        CascadeClassifier faceDetector = new CascadeClassifier(DEFAULT_PATH + "haarcascade_frontalface_my.xml");
        faceDetector.train(_cascadeDirName, _posFilename,
                _negFilename, _numPos, _numNeg, 
                _precalcValBufSize, _precalcIdxBufSize, 
                _numStages, _cascadeParams, _featureParams, _stageParams, baseFormatSave, acceptanceRatioBreakValue);*/
        
        return ;
    }
    
    
    
    /**
     * 加载样本文件
     * @param samples
     * @param labels
     * @param negative
     * @param positive
     */
    public static void loadSamples(Mat samples, Vector<Integer> labels, String negative, String positive) {
        // 加载负样本
        Vector<String> files = new Vector<String>();
        FileUtil.getFiles(negative, files);
        for (String img : files) {
            Mat inMat = Imgcodecs.imread(img);
            samples.push_back(inMat);
            labels.add(0);
        }
        // 加载正样本
        files = new Vector<String>();
        FileUtil.getFiles(positive, files);
        for (String img : files) {
            Mat inMat = Imgcodecs.imread(img);
            samples.push_back(inMat);
            labels.add(1);
        }
    }

    
    /**
     * 预测图片中是否包含人脸
     * 加载训练完成的模型文件，识别测试图片中是否包含人脸
     */
    public static void predict() {
        
    }
    

    /**
     * 从图片中，提取人脸图块(正样本)，或者疑似人脸的非人脸图块(负样本)
     * 将图块处理为相同大小正负样本文件，放到样本训练目录下
     * @param sourcePath 原图目录
     * @param targetPath 样本存放目录
     * @param limit 提取样本数量
     */
    public static void prepareSamples(String sourcePath, String targetPath, Integer limit) {
        Vector<String> files = new Vector<String>();
        FileUtil.getFiles(sourcePath, files);
        CascadeClassifier faceDetector = new CascadeClassifier(Constant.DEFAULT_FACE_MODEL_PATH);
        int i = 0;
        for (String img : files) {
            Mat inMat = Imgcodecs.imread(img);
            if(inMat.empty()) {
                continue;
            }
            Mat gray = ImageUtil.gray(inMat, false, "");
            Mat gsMat = ImageUtil.gaussianBlur(gray, false, "");
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(gsMat, faceDetections);
            for (Rect rect : faceDetections.toArray()) {
                // 截取人脸  灰度图
                Mat face = new Mat();
                Size size = new Size(rect.width, rect.height);
                Point center = new Point(rect.x + rect.width/2, rect.y + rect.height/2);
                Imgproc.getRectSubPix(gsMat, size, center, face);

                // resize 24*24
                Mat resized = new Mat(24, 24, CvType.CV_8UC3);
                Imgproc.resize(face, resized, resized.size(), 0, 0, Imgproc.INTER_CUBIC);
                // 保存文件
                Imgcodecs.imwrite(targetPath + getId() + ".jpg", resized);
                
                i++;
                if(i >= 2) {
                    return;
                }
            }
        }
        return;
    }


    public static void main(String[] args) {

        /*String sourcePath = "D:\\FaceDetect\\samples\\asia_all\\";
        String targetPath = "D:\\FaceDetect\\samples\\positive\\";
        // prepareSamples(sourcePath, targetPath, 4000);

        sourcePath = "D:\\FaceDetect\\samples\\europe_male\\";
        targetPath = "D:\\FaceDetect\\samples\\positive\\";
        // prepareSamples(sourcePath, targetPath, 2000);

        sourcePath = "D:\\FaceDetect\\samples\\europe_female\\";
        targetPath = "D:\\FaceDetect\\samples\\positive\\";
        // prepareSamples(sourcePath, targetPath, 2000);

        sourcePath = "D:\\PlateDetect\\plate\\";
        targetPath = "D:\\FaceDetect\\samples\\negative\\";
        // prepareSamples(sourcePath, targetPath, 4000);

        sourcePath = "D:\\PlateDetect\\plate1\\";
        targetPath = "D:\\FaceDetect\\samples\\negative\\";
        // prepareSamples(sourcePath, targetPath, 4000);
        
        train();
        
        predict();*/
        
        return;
    }

}
