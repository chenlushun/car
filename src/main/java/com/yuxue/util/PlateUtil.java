package com.yuxue.util;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.SVM;
import org.springframework.stereotype.Component;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.yuxue.constant.Constant;
import com.yuxue.enumtype.Direction;
import com.yuxue.enumtype.PlateColor;
import com.yuxue.train.SVMTrain;


/**
 * 车牌处理工具类
 * 车牌切图按字符分割
 * 字符识别
 * @author yuxue
 * @date 2020-05-28 15:11
 */
@Component
public class PlateUtil {

    private static String DEFAULT_BASE_TEST_PATH = "D:/PlateDetect/temp/";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // 这个位置加载模型文件会报错，暂时没时间定位啥问题报错
        /*loadSvmModel("D:/PlateDetect/train/plate_detect_svm/svm2.xml");
        loadAnnModel("D:/PlateDetect/train/chars_recognise_ann/ann.xml");
        loadAnnCnModel("D:/PlateDetect/train/chars_recognise_ann/ann_cn.xml");*/
    }

    PlateUtil(){
        loadSvmModel(Constant.DEFAULT_SVM_PATH);
        loadAnnModel(Constant.DEFAULT_ANN_PATH);
        loadAnnCnModel(Constant.DEFAULT_ANN_CN_PATH);
    }

    private static SVM svm = SVM.create();

    private static ANN_MLP ann=ANN_MLP.create();
    private static ANN_MLP ann_cn=ANN_MLP.create();

    public static void loadSvmModel(String path) {
        svm.clear();
        svm=SVM.load(path);
    }

    public static void loadAnnModel(String path) {
        ann.clear();
        ann = ANN_MLP.load(path);
    }

    public static void loadAnnCnModel(String path) {
        ann_cn.clear();
        ann_cn = ANN_MLP.load(path);
    }


    /**
     * 根据正则表达式判断字符串是否是车牌
     * @param str
     * @return
     */
    public static Boolean isPlate(String str) {
        Pattern p = Pattern.compile(Constant.plateReg);
        Boolean bl = false;
        Matcher m = p.matcher(str);
        while(m.find()) {
            bl = true;
            break;
        }
        return bl;
    }


    /**
     * 根据图片，获取可能是车牌的图块集合
     * @param src 输入原图
     * @param dst 可能是车牌的图块集合
     * @param debug 是否保留图片的处理过程
     * @param tempPath 图片处理过程的缓存目录
     */
    public static void getPlateMat(String imagePath, Vector<Mat> dst, Boolean debug, String tempPath) {
        Mat src = Imgcodecs.imread(imagePath);
        getPlateMat(src, dst, debug, tempPath);
    }

    /**
     * 根据图片，获取可能是车牌的图块集合
     * @param src 输入原图
     * @param dst 可能是车牌的图块集合
     * @param debug 是否保留图片的处理过程
     * @param tempPath 图片处理过程的缓存目录
     */
    public static void getPlateMat(Mat src, Vector<Mat> dst, Boolean debug, String tempPath) {
        // 灰度图
        Mat gray = ImageUtil.gray(src, debug, tempPath);

        // 高斯模糊
        Mat gsMat = ImageUtil.gaussianBlur(gray, debug, tempPath);

        // 颜色范围提取
        // Mat crMat = ImageUtil.getByColor(gsMat, debug, tempPath);

        // Sobel 运算，得到图像的一阶水平方向导数
        Mat sobel = ImageUtil.sobel(gsMat, debug, tempPath);

        // Mat scharr = ImageUtil.scharr(gray, debug, tempPath);    // 得到图像的一阶水平方向导数
        // Mat threshold = ImageUtil.threshold(scharr, debug, tempPath);

        // 图像进行二值化
        Mat threshold = ImageUtil.threshold(sobel, debug, tempPath);

        // 使用闭操作  同时处理一些干扰元素
        Mat morphology = ImageUtil.morphology(threshold, debug, tempPath);
        Mat clear1 = ImageUtil.clearInnerHole(morphology, 8, 16, debug, tempPath);
        Mat clear2 = ImageUtil.clearSmallConnArea(clear1, 4, 10, debug, tempPath);

        // Mat clear3 = ImageUtil.clearAngleConn(clear2, 5, debug, tempPath);

        // 获取图中所有的轮廓
        List<MatOfPoint> contours = ImageUtil.contours(src, clear2, debug, tempPath);
        // 根据轮廓， 筛选出可能是车牌的图块
        Vector<Mat> inMat = ImageUtil.screenBlock(src, contours, debug, tempPath);

        // 找出可能是车牌的图块，存到dst中， 返回结果
        hasPlate(inMat, dst, debug, tempPath);
    }


    /**
     * 输入车牌切图集合，判断是否包含车牌
     * @param inMat
     * @param dst 包含车牌的图块
     */
    public static void hasPlate(Vector<Mat> inMat, Vector<Mat> dst, Boolean debug, String tempPath) {
        int i = 0;
        for (Mat src : inMat) {
            if(src.rows() == Constant.DEFAULT_HEIGHT && src.cols() == Constant.DEFAULT_WIDTH) {
                Mat samples = SVMTrain.getFeature(src);
                float flag = svm.predict(samples);
                if (flag == 0) {
                    dst.add(src);
                    if(debug) {
                        // System.err.println("目标符合");
                        Imgcodecs.imwrite(tempPath + Constant.debugMap.get("platePredict") + "_platePredict_" + i + ".png", src);
                    }
                    i++;
                } else {
                    // System.out.println("目标不符合");
                }
            } else {
                System.err.println("非法图块");
            }
        }
        return;
    }


    /**
     * 判断切图车牌颜色
     * @param inMat
     * @return
     */
    public static PlateColor getPlateColor(Mat inMat, Boolean adaptive_minsv, Boolean debug, String tempPath) {
        // 判断阈值
        final float thresh = 0.70f;
        if(colorMatch(inMat, PlateColor.GREEN, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.GREEN;
        }
        if(colorMatch(inMat, PlateColor.YELLOW, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.YELLOW;
        }
        if(colorMatch(inMat, PlateColor.BLUE, adaptive_minsv, debug, tempPath) > thresh) {
            return PlateColor.BLUE;
        }
        return PlateColor.UNKNOWN;
    }


    /**
     * 颜色匹配计算
     * @param inMat
     * @param r
     * @param adaptive_minsv
     * @param debug
     * @param tempPath
     * @return
     */
    public static Float colorMatch(Mat inMat, PlateColor r, Boolean adaptive_minsv, Boolean debug, String tempPath) {
        final float max_sv = 255;
        final float minref_sv = 64;
        final float minabs_sv = 95;

        Mat hsvMat = ImageUtil.rgb2Hsv(inMat, debug, tempPath);

        // 匹配模板基色,切换以查找想要的基色
        int min_h = r.minH;
        int max_h = r.maxH;
        float diff_h = (float) ((max_h - min_h) / 2);
        int avg_h = (int) (min_h + diff_h);

        for (int i = 0; i < hsvMat.rows(); ++i) {
            for (int j = 0; j < hsvMat.cols(); j += 3) {
                int H = (int)hsvMat.get(i, j)[0];
                int S = (int)hsvMat.get(i, j)[1];
                int V = (int)hsvMat.get(i, j)[2];

                boolean colorMatched = false;

                if ( min_h < H && H <= max_h) {
                    int Hdiff = Math.abs(H - avg_h);
                    float Hdiff_p = Hdiff / diff_h;
                    float min_sv = 0;
                    if (adaptive_minsv) {
                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p);
                    } else {
                        min_sv = minabs_sv;
                    }
                    if ((min_sv < S && S <= max_sv) && (min_sv < V && V <= max_sv)) {
                        colorMatched = true;
                    }
                }

                if (colorMatched == true) {
                    hsvMat.put(i, j, 0, 0, 255);
                } else {
                    hsvMat.put(i, j, 0, 0, 0);
                }
            }
        }

        // 获取颜色匹配后的二值灰度图
        List<Mat> hsvSplit = Lists.newArrayList();
        Core.split(hsvMat, hsvSplit);
        Mat gray = hsvSplit.get(2);

        float percent = (float) Core.countNonZero(gray) / (gray.rows() * gray.cols());
        if (debug) {
            Imgcodecs.imwrite(tempPath + Constant.debugMap.get("colorMatch") + "_colorMatch.jpg", gray);
        }
        return percent;
    }



    /**
     * 车牌切图，分割成单个字符切图
     * @param inMat 输入原始图像
     * @param charMat 返回字符切图vector
     * @param debug
     * @param tempPath
     */
    public static final int DEFAULT_ANGLE = 30; // 角度判断所用常量
    public static String charsSegment(Mat inMat, PlateColor color, Boolean debug, String tempPath) {
        // 切换到灰度图 
        Mat gray = new Mat();
        Imgproc.cvtColor(inMat, gray, Imgproc.COLOR_BGR2GRAY);

        // 图像进行二值化
        Mat threshold = new Mat();
        switch (color) {
        case BLUE:
            Imgproc.threshold(gray, threshold, 10, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);
            break;
        case YELLOW:
            Imgproc.threshold(gray, threshold, 10, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY_INV);
            break;
        case GREEN:
            Imgproc.threshold(gray, threshold, 10, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY_INV);
            break;
        default:
            return null;
        }
        if (debug) {    // 输出二值图
            Imgcodecs.imwrite(tempPath + Constant.debugMap.get("plateThreshold") + "_plateThreshold.jpg", threshold);
        }

        // 提取外部轮廓
        List<MatOfPoint> contours = Lists.newArrayList();
        Imgproc.findContours(threshold, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        if (debug) {    // 输出轮廓图
            Mat result = new Mat();
            inMat.copyTo(result);
            Imgproc.drawContours(result, contours, -1, new Scalar(0, 0, 255, 255));
            Imgcodecs.imwrite(tempPath + Constant.debugMap.get("plateContours") + "_plateContours.jpg", result);
        }

        Vector<Rect> rt = new Vector<Rect>();
        for (int i = 0; i < contours.size(); i++) {
            Rect mr = Imgproc.boundingRect(contours.get(i));    //  boundingRect()得到包覆此轮廓的最小正矩形

            if (checkCharSizes(mr)) {   // 验证尺寸，主要验证高度是否满足要求，去掉不符合规格的字符，中文字符后续处理
                rt.add(mr);
            }
        }
        if(null == rt || rt.size() <= 0) {  // 未识别到字符
            return null;
        }

        // 排序 
        Vector<Rect> sorted = new Vector<Rect>();
        sortRect(rt, sorted);

        // 定位省份字母位置
        Integer posi = getSpecificRect(sorted, color);

        // 定位中文字符等等
        Rect chineseRect = getChineseRect(sorted.get(posi));

        Mat chineseMat = new Mat(threshold, chineseRect);
        chineseMat = preprocessChar(chineseMat);
        if (debug) {  
            Imgcodecs.imwrite(tempPath + Constant.debugMap.get("chineseMat") + "_chineseMat.jpg", chineseMat);
        }

        String plate = "";

        // 预测中文字符
        plate = plate + predictChinese(chineseMat);

        // 预测中文之外的字符
        for (int i = 0; i < sorted.size(); i++) {
            if(i < posi) {
                continue;
            }
            int charCount = 7;
            if(color.equals(PlateColor.GREEN)) {
                charCount = 8;
            }
            if(i >= charCount) {
                continue;
            }
            Mat img_crop = new Mat(threshold, sorted.get(i));
            img_crop = preprocessChar(img_crop);
            plate = plate + predict(img_crop);
            if (debug) {  
                Imgcodecs.imwrite(tempPath + Constant.debugMap.get("specMat") + "_specMat_" + i + ".jpg", img_crop);
            }
        }
        System.err.println("=========>" + plate);
        return plate;
    }


    public static String predict(Mat img) {
        Mat f = PlateUtil.features(img, Constant.predictSize);

        int index = 0;
        double maxVal = -2;
        Mat output = new Mat(1, Constant.strCharacters.length, CvType.CV_32F);
        ann.predict(f, output);  // 预测结果
        for (int j = 0; j < Constant.strCharacters.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }

        // 膨胀
        f = PlateUtil.features(PlateUtil.dilate(img), Constant.predictSize);
        ann.predict(f, output);  // 预测结果
        for (int j = 0; j < Constant.strCharacters.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }

        String result = String.valueOf(Constant.strCharacters[index]);
        return result;
    }

    public static String predictChinese(Mat img) {
        Mat f = PlateUtil.features(img, Constant.predictSize);
        int index = 0;
        double maxVal = -2;

        Mat output = new Mat(1, Constant.strChinese.length, CvType.CV_32F);
        ann_cn.predict(f, output);  // 预测结果
        for (int j = 0; j < Constant.strChinese.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }
        // 腐蚀  -- 识别中文字符效果会好一点，识别数字及字母效果会更差
        f = PlateUtil.features(PlateUtil.erode(img), Constant.predictSize);
        ann_cn.predict(f, output);  // 预测结果
        for (int j = 0; j < Constant.strChinese.length; j++) {
            double val = output.get(0, j)[0];
            if (val > maxVal) {
                maxVal = val;
                index = j;
            }
        }
        String result = Constant.strChinese[index];
        return Constant.KEY_CHINESE_MAP.get(result);
    }


    /**
     * 找出指示城市的字符的Rect，例如 苏A7003X，就是A的位置
     * 之所以选择城市的字符位置，是因为该位置不管什么字母，占用的宽度跟高度的差不多，而且字符笔画是连续的，能大大提高位置的准确性
     * @param vecRect
     * @return
     */
    public static Integer getSpecificRect(Vector<Rect> vecRect, PlateColor color) {
        List<Integer> xpositions = Lists.newArrayList();

        int maxHeight = 0;
        int maxWidth = 0;
        for (int i = 0; i < vecRect.size(); i++) {
            xpositions.add(vecRect.get(i).x);
            if (vecRect.get(i).height > maxHeight) {
                maxHeight = vecRect.get(i).height;
            }
            if (vecRect.get(i).width > maxWidth) {
                maxWidth = vecRect.get(i).width;
            }
        }
        int specIndex = 0;
        for (int i = 0; i < vecRect.size(); i++) {
            Rect mr = vecRect.get(i);
            int midx = mr.x + mr.width / 2;

            if(PlateColor.GREEN.equals(color)) {
                if ((mr.width > maxWidth * 0.8 || mr.height > maxHeight * 0.8)
                        && (midx < Constant.DEFAULT_WIDTH * 2 / 8 && midx > Constant.DEFAULT_WIDTH / 8)) {
                    specIndex = i;
                }
            } else {
                // 如果一个字符有一定的大小，并且在整个车牌的1/7到2/7之间，则是我们要找的特殊车牌
                if ((mr.width > maxWidth * 0.8 || mr.height > maxHeight * 0.8)
                        && (midx < Constant.DEFAULT_WIDTH * 2 / 7 && midx > Constant.DEFAULT_WIDTH / 7)) {
                    specIndex = i;
                } 
            }
        }
        return specIndex;
    }


    /**
     * 根据特殊车牌来构造猜测中文字符的位置和大小
     * 
     * @param rectSpe
     * @return
     */
    public static Rect getChineseRect(Rect rectSpe) {
        int height = rectSpe.height;
        float newwidth = rectSpe.width * 1.15f;
        int x = rectSpe.x;
        int y = rectSpe.y;

        int newx = x - (int) (newwidth * 1.15);
        newx = Math.max(newx, 0);
        Rect a = new Rect(newx, y, (int) newwidth, height);
        return a;
    }

    /**
     * 这个函数做两个事情
     * <ul>
     * <li>把特殊字符Rect左边的全部Rect去掉，后面再重建中文字符的位置;
     * <li>从特殊字符Rect开始，依次选择6个Rect，多余的舍去。
     * <ul>
     * @return
     */
    private int RebuildRect(final Vector<Rect> vecRect, Vector<Rect> outRect, int specIndex, PlateColor color) {
        // 最大只能有7个Rect,减去中文的就只有6个Rect
        int count = 6;
        if(PlateColor.GREEN.equals(color)) {
            count = 7; // 绿牌要多一个
        }
        for (int i = 0; i < vecRect.size(); i++) {
            // 将特殊字符左边的Rect去掉，这个可能会去掉中文Rect，不过没关系，我们后面会重建。
            if (i < specIndex)
                continue;

            outRect.add(vecRect.get(i));
            if (--count == 0)
                break;
        }
        return 0;
    }


    /**
     * 字符预处理: 统一每个字符的大小
     * @param in
     * @return
     */
    final static int CHAR_SIZE = 20;
    private static Mat preprocessChar(Mat in) {
        int h = in.rows();
        int w = in.cols();
        Mat transformMat = Mat.eye(2, 3, CvType.CV_32F);
        int m = Math.max(w, h);
        transformMat.put(0, 2, (m - w) / 2f);
        transformMat.put(1, 2, (m - h) / 2f);

        Mat warpImage = new Mat(m, m, in.type());
        Imgproc.warpAffine(in, warpImage, transformMat, warpImage.size(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0));

        Mat resized = new Mat(CHAR_SIZE, CHAR_SIZE, CvType.CV_8UC3);
        Imgproc.resize(warpImage, resized, resized.size(), 0, 0, Imgproc.INTER_CUBIC);

        return resized;
    }


    /**
     * 字符尺寸验证；去掉尺寸不符合的图块
     * 此处计算宽高比意义不大，因为字符 1 的宽高比干扰就已经很大了
     * @param r
     * @return
     */
    public static Boolean checkCharSizes(Rect r) {
        float minHeight = 15f;
        float maxHeight = 35f;
        double charAspect = r.size().width / r.size().height;
        return charAspect <1 && minHeight <= r.size().height && r.size().height < maxHeight;
    }



    /**
     * 将Rect按位置从左到右进行排序
     * @param vecRect
     * @param out
     * @return
     */
    public static void sortRect(Vector<Rect> vecRect, Vector<Rect> out) {
        Map<Integer, Integer> map = Maps.newHashMap();
        for (int i = 0; i < vecRect.size(); ++i) {
            map.put(vecRect.get(i).x, i);
        }
        Set<Integer> set = map.keySet();
        Object[] arr = set.toArray();
        Arrays.sort(arr);
        for (Object key : arr) {
            out.add(vecRect.get(map.get(key)));
        }
        return;
    }


    public static float[] projectedHistogram(final Mat img, Direction direction) {
        int sz = 0;
        switch (direction) {
        case HORIZONTAL:
            sz = img.rows();
            break;

        case VERTICAL:
            sz = img.cols();
            break;

        default:
            break;
        }
        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
        float[] nonZeroMat = new float[sz];
        Core.extractChannel(img, img, 0);
        for (int j = 0; j < sz; j++) {
            Mat data = (direction == Direction.HORIZONTAL) ? img.row(j) : img.col(j);
            int count = Core.countNonZero(data);
            nonZeroMat[j] = count;
        }
        float max = 0;
        for (int j = 0; j < nonZeroMat.length; ++j) {
            max = Math.max(max, nonZeroMat[j]);
        }
        if (max > 0) {
            for (int j = 0; j < nonZeroMat.length; ++j) {
                nonZeroMat[j] /= max;
            }
        }
        return nonZeroMat;
    }


    public static Mat features(Mat in, int sizeData) {
        float[] vhist = projectedHistogram(in, Direction.VERTICAL);
        float[] hhist = projectedHistogram(in, Direction.HORIZONTAL);
        Mat lowData = new Mat();
        if (sizeData > 0) {
            Imgproc.resize(in, lowData, new Size(sizeData, sizeData));
        }
        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = new Mat(1, numCols, CvType.CV_32F);

        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            out.put(0, j, vhist[i]);
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            out.put(0, j, hhist[i]);
        }
        for (int x = 0; x < lowData.cols(); x++) {
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                double[] val = lowData.get(x, y);
                out.put(0, j, val[0]);
            }
        }
        return out;
    }



    /**
     * 进行膨胀操作
     * 也可以理解为字体加粗操作
     * @param inMat
     * @return
     */
    public static Mat dilate(Mat inMat) {
        Mat result = inMat.clone();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.dilate(inMat, result, element);
        return result;
    }

    /**
     * 进行腐蚀操作
     * @param inMat
     * @return
     */
    public static Mat erode(Mat inMat) {
        Mat result = inMat.clone();
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Imgproc.erode(inMat, result, element);
        return result;
    }


    /**
     * 随机数平移
     * @param inMat
     * @return
     */
    public static Mat randTranslate(Mat inMat) {
        Random rand = new Random();
        Mat result = inMat.clone();
        int ran_x = rand.nextInt(10000) % 5 - 2; // 控制在-2~3个像素范围内
        int ran_y = rand.nextInt(10000) % 5 - 2;
        return translateImg(result, ran_x, ran_y);
    }


    /**
     * 随机数旋转
     * @param inMat
     * @return
     */
    public static Mat randRotate(Mat inMat) {
        Random rand = new Random();
        Mat result = inMat.clone();
        float angle = (float) (rand.nextInt(10000) % 15 - 7); // 旋转角度控制在-7~8°范围内
        return rotateImg(result, angle);
    }


    /**
     * 平移
     * @param img
     * @param offsetx
     * @param offsety
     * @return
     */
    public static Mat translateImg(Mat img, int offsetx, int offsety){
        Mat dst = new Mat();
        //定义平移矩阵
        Mat trans_mat = Mat.zeros(2, 3, CvType.CV_32FC1);
        trans_mat.put(0, 0, 1);
        trans_mat.put(0, 2, offsetx);
        trans_mat.put(1, 1, 1);
        trans_mat.put(1, 2, offsety);
        Imgproc.warpAffine(img, dst, trans_mat, img.size());    // 仿射变换
        return dst;
    }


    /**
     * 旋转角度
     * @param source
     * @param angle
     * @return
     */
    public static Mat rotateImg(Mat source, float angle){
        Point src_center = new Point(source.cols() / 2.0F, source.rows() / 2.0F);
        Mat rot_mat = Imgproc.getRotationMatrix2D(src_center, angle, 1);
        Mat dst = new Mat();
        // 仿射变换 可以考虑使用投影变换; 这里使用放射变换进行旋转，对于实际效果来说感觉意义不大，反而会干扰结果预测
        Imgproc.warpAffine(source, dst, rot_mat, source.size());    
        return dst;
    }


    public static void main(String[] args) {
        // main方法执行，会调用PlateUtil的static方法，但是加载xml文件放在static，会报异常
        // 通过spring管理对象，会先执行static，然后执行构造方法
        PlateUtil.loadSvmModel(Constant.DEFAULT_DIR + "train/plate_detect_svm/svm2.xml");
        PlateUtil.loadAnnModel(Constant.DEFAULT_DIR + "train/chars_recognise_ann/ann2.xml");
        PlateUtil.loadAnnCnModel(Constant.DEFAULT_DIR + "train/chars_recognise_ann/ann_cn.xml");

        Instant start = Instant.now();
        String tempPath = DEFAULT_BASE_TEST_PATH + "test/";
        String filename = tempPath + "/100_yuantu1.jpg";

        Boolean debug = true;
        Vector<Mat> dst = new Vector<Mat>();
        getPlateMat(filename, dst, debug, tempPath);

        System.err.println("识别到的车牌数量：" + dst.size());

        dst.stream().forEach(inMat -> {
            PlateColor color = PlateUtil.getPlateColor(inMat, debug, debug, tempPath);
            System.err.println(color.desc);
            PlateUtil.charsSegment(inMat, color, debug, tempPath);
        });

        Instant end = Instant.now();
        System.err.println("总耗时：" + Duration.between(start, end).toMillis());
    }

}
