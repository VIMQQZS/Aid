/*
 * @Author: your name
 * @Date: 2021-05-25 14:53:26
 * @LastEditTime: 2021-05-25 14:55:05
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \\undefinedc:\\Users\\Vim\\Desktop\\VScode\\Java\\src\\Contacts\\FileRW.java
 */
package contacts;

import java.io.*;

/**
 * Created by Yifan Jia on 2018/6/10.
 */
public class FileRW {
    private static FileWriter fileWriter;

    private static FileReader fileReader;

    private static BufferedReader bf;

    private static BufferedWriter bw;

    private static File file = new File("D:\\dest.txt");

    public static void fileWrite(String s) {
        try {
            fileWriter = new FileWriter(file, true);
            bw = new BufferedWriter(fileWriter);
            bw.write(s);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                bw.close();
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static String fileRead(String dest) {
        try {
            fileReader = new FileReader(file);
            bf = new BufferedReader(fileReader);
            String ss;
            while ((ss = bf.readLine()) != null) {
                String[] temp = ss.split(",");
                if (temp[0].equals(dest)) {
                    return ss;
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                bf.close();
                fileReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
}