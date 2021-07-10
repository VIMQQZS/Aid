
/*
 * @Author: your name
 * @Date: 2021-05-14 12:16:48
 * @LastEditTime: 2021-05-14 12:43:49
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \\RM_Vision\\Java\\Hello\\src\\wordvec.java
 */
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class wordvec {
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader("E:\\IDEA_2018\\BigData\\data\\CopyDemo\\Demo.txt"));// 选择要读取要进行统计字符的文件的路径
        BufferedWriter bw = new BufferedWriter(
                new FileWriter("E:\\IDEA_2018\\BigData\\data\\CopyDemo\\Demo02\\ddd.txt"));// 将最后统计的结果写入文件的路径
        int i;
        StringBuilder sb = new StringBuilder();// 定义一个StringBuilder 变量用来接收读取的内容
        HashMap<String, Integer> map = new HashMap<>();// 初始化一个HasmMap用来存储读取的单词
        char[] bytes = new char[1024];
        while ((i = br.read(bytes)) != -1) {// 开始读取文件
            sb.append(new String(bytes, 0, i));// 将读取的内容添加到sb中
        }
        br.close();
        String[] split = sb.toString().split("\r\n\\u002E\\u003F\\,");// 将sb 转化为String类型的 按照换行符进行切分存入字符串数组中

        for (String s : split) { // 开始遍历上面的字符串数组，每遍历一次将s 作为k添加到map中
            map.put(s, 0);
        }
        Set<Map.Entry<String, Integer>> entries = map.entrySet();
        for (String s : split// 使用二重循环 统计单词出现的次数
        ) {
            for (Map.Entry<String, Integer> entry : entries) {
                if (s.equals(entry.getKey())) {
                    entry.setValue(entry.getValue() + 1);
                }
            }
        }
        for (Map.Entry<String, Integer> entry : entries) {// 遍历HashMap 将map中的内容写入到文件中
            bw.write(entry.getKey() + " " + entry.getValue());
            bw.write("\n");
            bw.flush();
        }
        bw.close();
    }
}
