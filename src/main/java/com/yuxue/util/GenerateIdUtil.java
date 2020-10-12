package com.yuxue.util;



/**
 * 用于生成ID的工具类
 * @author yuxue
 * @date 2020-10-12 11:16
 */
public class GenerateIdUtil {

    /**
     * 获取时间戳，生成文件名称
     * @return
     */
    public static synchronized Long getId() {
        try {
            Thread.sleep(1);
        } catch (Exception e) {}
        return System.currentTimeMillis();
    }
    
    
    /**
     * 获取时间戳，生成文件名称
     * @return
     */
    public static synchronized String getStrId() {
        try {
            Thread.sleep(1);
        } catch (Exception e) {}
        return System.currentTimeMillis() + "";
    }

    
    
}
