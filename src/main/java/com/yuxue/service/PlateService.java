package com.yuxue.service;


public interface PlateService {
    
    public Object getProcessStep();
    
    public Object recognise(String filePath, boolean reRecognise);

    public Object refreshFileInfo();
    
    public Object recogniseAll();
    
    public Object getImgInfo(String imgPath, Integer rows, Integer cols);
    
    public Object getHsvValue(String imgPath, Integer rows, Integer cols);
    
    
    
}