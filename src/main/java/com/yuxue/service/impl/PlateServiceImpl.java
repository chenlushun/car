package com.yuxue.service.impl;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import org.opencv.core.Mat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

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
        Random r = new Random(99);
        list.parallelStream().forEach(n->{
            File f = new File(n.getFilePath());
            if(FileUtil.checkFile(f)) {
                doRecognise(f, n, r.nextInt());
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
            doRecognise(f, e, 0); // 重新识别
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
    public Object doRecognise(File f, PlateFileEntity e, Integer seed) {

        Long ct = System.currentTimeMillis();

        // 先将文件拷贝并且重命名到不包含中文及特殊字符的目录下
        String targetPath = Constant.DEFAULT_TEMP_DIR.concat(ct.toString() + seed)
                .concat(f.getAbsolutePath().substring(f.getAbsolutePath().lastIndexOf(".")));
        FileUtil.copyAndRename(f.getAbsolutePath(), targetPath);

        // 创建临时目录， 存放过程图片
        String tempPath =  Constant.DEFAULT_TEMP_DIR.concat(ct.toString() + seed).concat("/");
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


}
