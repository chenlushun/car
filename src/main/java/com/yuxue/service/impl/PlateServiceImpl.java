package com.yuxue.service.impl;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.alibaba.fastjson.JSONObject;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.yuxue.constant.Constant;
import com.yuxue.entity.PlateFileEntity;
import com.yuxue.entity.PlateRecoDebugEntity;
import com.yuxue.entity.TempPlateFileEntity;
import com.yuxue.enumtype.PlateColor;
import com.yuxue.mapper.PlateFileMapper;
import com.yuxue.mapper.PlateRecoDebugMapper;
import com.yuxue.mapper.TempPlateFileMapper;
import com.yuxue.service.PlateService;
import com.yuxue.util.FileUtil;
import com.yuxue.util.PlateUtil;


@Service
public class PlateServiceImpl implements PlateService {

    @Autowired
    private PlateFileMapper plateFileMapper;

    @Autowired
    private PlateRecoDebugMapper plateRecoDebugMapper;

    @Autowired
    private TempPlateFileMapper tempPlateFileMapper;
    
    
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    
    /**
     * 获取时间戳，生成文件名称
     * @return
     */
    public static synchronized Long getId() {
        return System.currentTimeMillis();
    }


    @Override
    @Transactional(propagation = Propagation.REQUIRED)
    public Object refreshFileInfo() {
        File baseDir = new File(Constant.DEFAULT_DIR);
        if(!baseDir.exists() || !baseDir.isDirectory()) {
            return null;
        }
        List<TempPlateFileEntity> resultList = Lists.newArrayList();

        // 获取baseDir下第一层级的目录， 仅获取文件夹，不递归子目录，遍历
        List<File> folderList = FileUtil.listFile(baseDir, ";", false);
        folderList.parallelStream().forEach(folder -> {
            if(!folder.getName().equals("temp")) {
                // 遍历每一个文件夹， 递归获取文件夹下的图片
                List<File> imgList = FileUtil.listFile(folder, Constant.DEFAULT_TYPE, true);
                if(null != imgList && imgList.size() > 0) {
                    imgList.parallelStream().forEach(n->{
                        TempPlateFileEntity entity = new TempPlateFileEntity();
                        entity.setFilePath(n.getAbsolutePath().replaceAll("\\\\", "/"));
                        entity.setFileName(n.getName());
                        entity.setFileType(n.getName().substring(n.getName().lastIndexOf(".") + 1));
                        resultList.add(entity);
                    });
                }
            }
        });

        tempPlateFileMapper.turncateTable();
        tempPlateFileMapper.batchInsert(resultList);
        tempPlateFileMapper.updateFileInfo();
        return 1;
    }


    @Override
    public Object recogniseAll() {
        // 查询到还没有进行车牌识别的图片
        List<PlateFileEntity> list = plateFileMapper.getUnRecogniseList();

        // 开启多线程进行识别
        list.parallelStream().forEach(n->{
            File f = new File(n.getFilePath());
            if(FileUtil.checkFile(f)) {
                doRecognise(f, n);
            }
        });

        return 1;
    }


    @Override
    public Object getProcessStep() {
        return Constant.debugMap;
    }


    @Override
    public Object recognise(String filePath, boolean reRecognise) {
        filePath = filePath.replaceAll("\\\\", "/");
        File f = new File(filePath);
        PlateFileEntity e = null;

        Map<String, Object> paramMap = Maps.newHashMap();
        paramMap.put("filePath", filePath);
        List<PlateFileEntity> list= plateFileMapper.selectByCondition(paramMap);
        if(null == list || list.size() <= 0) {
            if(FileUtil.checkFile(f)) {
                e = new PlateFileEntity();
                e.setFileName(f.getName());
                e.setFilePath(f.getAbsolutePath().replaceAll("\\\\", "/"));
                e.setFileType(f.getName().substring(f.getName().lastIndexOf(".") + 1));
                plateFileMapper.insertSelective(e);
            }
            reRecognise = true;
        } else {
            e = list.get(0);
        }

        if(reRecognise) {
            doRecognise(f, e); // 重新识别
            e = plateFileMapper.selectByPrimaryKey(e.getId()); // 重新识别之后，重新获取一下数据
        }

        // 查询数据库，返回结果
        paramMap.clear();
        paramMap.put("parentId", e.getId());
        e.setDebug(plateRecoDebugMapper.selectByCondition(paramMap));

        return e;
    }

    /**
     * 单张图片 车牌识别
     * 拷贝文件到临时目录
     * 过程及结果更新数据库
     * @param f 调用方需要验证文件存在
     * @param result
     * @return
     */
    public Object doRecognise(File f, PlateFileEntity e) {

        Long ct = getId();

        // 先将文件拷贝并且重命名到不包含中文及特殊字符的目录下
        String targetPath = Constant.DEFAULT_TEMP_DIR.concat(ct.toString())
                .concat(f.getAbsolutePath().substring(f.getAbsolutePath().lastIndexOf(".")));
        FileUtil.copyAndRename(f.getAbsolutePath(), targetPath);

        // 创建临时目录， 存放过程图片
        String tempPath =  Constant.DEFAULT_TEMP_DIR.concat(ct.toString()).concat("/");
        FileUtil.createDir(tempPath); // 创建文件夹

        Boolean debug = true;
        Vector<Mat> dst = new Vector<Mat>();
        PlateUtil.getPlateMat(targetPath, dst, debug, tempPath);

        Set<String> plates = Sets.newHashSet();
        Set<String> colors = Sets.newHashSet();
        dst.stream().forEach(inMat -> {
            PlateColor color = PlateUtil.getPlateColor(inMat, debug, debug, tempPath);
            colors.add(color.code);
            String plate = PlateUtil.charsSegment(inMat, color, debug, tempPath);
            plates.add(plate);
        });

        e.setTempPath(tempPath);
        e.setRecoColor(colors.toString());
        e.setRecoPlate(plates.toString());

        // 删除拷贝的文件
        new File(targetPath).delete();
        
        // 插入识别过程图片数据信息 通过temp文件夹的文件，更新数据库
        List<PlateRecoDebugEntity> list = Lists.newArrayList();
        List<File> debugList = FileUtil.listFile(new File(tempPath), Constant.DEFAULT_TYPE, false);
        debugList.parallelStream().forEach(d -> {
            String fileName = d.getName();
            
            String debugType = fileName.substring(fileName.indexOf("_") + 1, fileName.lastIndexOf("."));
            if(debugType.contains("_")) {
                debugType = debugType.substring(0, debugType.lastIndexOf("_"));
            }

            PlateRecoDebugEntity de = new PlateRecoDebugEntity();
            de.setRecoPlate("");
            de.setPlateColor("");
            de.setFilePath(d.getAbsolutePath().replaceAll("\\\\", "/"));
            de.setFileName(d.getName());
            de.setParentId(e.getId());
            de.setDebugType(debugType);
            de.setSort(Constant.debugMap.get(debugType));
            list.add(de);
        });

        // 更新图片主表信息
        plateFileMapper.updateByPrimaryKeySelective(e);
        plateRecoDebugMapper.deleteByParentId(e.getId());
        plateRecoDebugMapper.batchInsert(list);

        return 1;
    }

    @Override
    public Object getImgInfo(String imgPath) {
        Map<String, Object> result = Maps.newHashMap();
        
        Long ct = getId();

        File f = new File(imgPath);
        if(f.exists()) {
            // 先将文件拷贝并且重命名到不包含中文及特殊字符的目录下
            String targetPath = Constant.DEFAULT_TEMP_DIR.concat(ct.toString())
                    .concat(f.getAbsolutePath().substring(f.getAbsolutePath().lastIndexOf(".")));
            FileUtil.copyAndRename(f.getAbsolutePath(), targetPath);
            result.put("targetPath", targetPath);   // 返回临时路径给前端
            // 获取图片的基本信息
            Mat inMat = Imgcodecs.imread(targetPath);
            result.put("rows", inMat.rows());
            result.put("cols", inMat.cols());
        }
        return result;
    }

    
    @Override
    public Object getHSVValue(String imgPath, Integer row, Integer col) {
        Map<String, Object> result = Maps.newHashMap();
        Mat inMat = Imgcodecs.imread(imgPath);

        double[] rgb = inMat.get(row, col);
        result.put("RGB", JSONObject.toJSONString(rgb));

        Mat dst = new Mat(inMat.rows(), inMat.cols(), CvType.CV_32FC3);
        Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2HSV); // 转到HSV空间进行处理

        double[] hsv = dst.get(row, col);
        result.put("HSV", JSONObject.toJSONString(hsv));
        return result;
    }
    

    public static void main(String[] args) {
        Mat inMat = Imgcodecs.imread("D:\\PlateDetect\\temp\\test\\qietu.png");
        Mat dst = new Mat(inMat.rows(), inMat.cols(), CvType.CV_32FC3);
        Imgproc.cvtColor(inMat, dst, Imgproc.COLOR_BGR2HSV); // 转到HSV空间进行处理
        double[] d = null;
        for (int row = 0; row < inMat.rows(); row++) {
            for (int col = 0; col < inMat.cols(); col++) {
                d = dst.get(row, col);
                String s = JSONObject.toJSONString(d);
                if(!s.equals("[0.0,0.0,0.0]")) {    // png图片，透明部分颜色取值：[0.0,0.0,0.0]
                    System.out.print(row);
                    System.out.print("\t" + col);
                    System.out.println("\t" + s);
                }
            }
        }
        return;
    }


}
