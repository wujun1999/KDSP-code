package tool;

import java.io.File;

import java.io.FileOutputStream;

import java.io.IOException;

import java.io.OutputStream;

public class LogWrite3 {
	private static boolean fileLog = true;
	private static String logFileName = "D:\\work\\MetaNN-master_2\\MetaNN-master\\DataSet\\D007410\\Data\\MF_1_5\\j14\\final.txt";// 指定程序执行结果保存的文件路径

	public static OutputStream getOutputStream() throws IOException {
		if (fileLog) {
			File file = new File(logFileName);
			if (!file.exists())
				file.createNewFile();
			return new FileOutputStream(file, true);
		} else {
			return System.out;
		}
	}

	public static void log(String info) throws IOException {
		OutputStream out = getOutputStream();
		out.write(info.getBytes("utf-8"));
	}
}
