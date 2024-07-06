package datamodel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import tool.SimpleTool;
import java.io.FileOutputStream;  
import java.io.PrintStream;

public class DataInfo {
	public static int userNumber = 766;// jester1:24983; jester2:23500; jester3:24938     // 行数   168
	public static int itemNumber = 144;    // 列数101   58
	public static int rateNumber = 110304;// jester1:1810455; jester2:1708993; jester3:616912   // 单元格个数   第二个：2373500   9744
	public static int trNumber = 0;
	public static int teNumber = 0;
	public static int teIndxRem = 4;// 0~9取任意值
	public static int theta = 1;//大于theta，进行推荐

	/********************** Feature Matrix ***********************************/
	public static short featureNumber = 10;
	public static double[][] uFeature = new double[userNumber][featureNumber];
	public static double[][] iFeature = new double[itemNumber][featureNumber];

	/******************** Training set *******************************/
	public static Triple[] data = new Triple[rateNumber];
	public static double[] uTrTotalRates = new double[userNumber];
	public static double[] uTrAverRates = new double[userNumber];
	public static int[] uTrTotalNum = new int[userNumber];
	public static double[] iTrTotalRates = new double[itemNumber];
	public static double[] iTrAverRates = new double[itemNumber];
	public static int[] iTrTotalNum = new int[itemNumber];

	public static int round = 1000;
	public static double mean_rating = 0;
	
	public static String dataPath = new String("data/12891.txt");
	static String split_Sign = new String(" ");          //分割字符串 用空格分割
	//static String split_Sign = new String("	");

	static int[] userCount;
	static int[] userPot;

	public DataInfo(String paraDataPath) throws IOException {
		readData(paraDataPath);
		setPot();
	}// of the first constructor

	/**
	 * 
	 * @param paraDataPath
	 * @throws IOException
	 */
	static void readData(String paraDataPath) throws IOException {
		
		File file = new File(paraDataPath);
		BufferedReader buffRead = new BufferedReader
				                 (new InputStreamReader(new FileInputStream(file)));

		double sum = 0;
		int userIndex = 0;
		int index = 0;
		
		for (int i = 0; i < DataInfo.rateNumber; i++)
			data[i] = new Triple();
		
		while (buffRead.ready()) {                         // 读文件
			String str = buffRead.readLine();
//			String[] parts = str.split(" ");
			String[] parts = str.split("	");
			
			int user = userIndex;// user id
			for (int i = 1; i < itemNumber; i++) {
				int item = i - 1;// item id
				double rating = Double.parseDouble(parts[i]);// 将rating转为double型储存
				
				data[index].user = user;
				data[index].item = item;
				data[index].rate = rating;
				if (index % 10 != teIndxRem) {
					sum += rating;// total training rating
					uTrTotalRates[user] += rating;
					uTrTotalNum[user]++;
					iTrTotalRates[item] += rating;
					iTrTotalNum[item]++;
					trNumber++;}
				index++;

//				if (rating != 99) { //只选择有评分数据的user和item
//					data[index].user = user;         // 第index三元组里面的user
//					data[index].item = item;
//					data[index].rate = rating;
//					if (index % 10 != teIndxRem) {        // 意义是什么？
//						sum += rating;// total training rating
//						uTrTotalRates[user] += rating;
//						uTrTotalNum[user]++;
//						iTrTotalRates[item] += rating;
//						iTrTotalNum[item]++;
//						trNumber++;
//					} // Of if
//					index++;
//				} // Of if
				
			} // Of for i
			userIndex++;
		} // Of while

		teNumber = rateNumber - trNumber;      // 在这里是得到值不是99的评分的个数 我用不到
		System.out.println("index:" + index);    // 输出评分值不是99的个数
		mean_rating = sum / trNumber;   // average rating
		
		// Compute average rating for each user
		for (int i = 0; i < userNumber; i++)
			if (uTrTotalNum[i] > 1e-6) {
				uTrAverRates[i] = uTrTotalRates[i] / uTrTotalNum[i];
			} // of if

		// Compute average rating for each user
		for (int i = 0; i < itemNumber; i++) 
			if (iTrTotalNum[i] > 1e-6) {
				iTrAverRates[i] = iTrTotalRates[i] / iTrTotalNum[i];
			} // of if

		for (int i = 0; i < DataInfo.rateNumber; i++) {
			double tmp = (Double) data[i].rate;
			data[i].rate = tmp;
		} // of for i
		buffRead.close();
	}

	static void setPot() {

		userCount = new int[DataInfo.userNumber + 1];
		userPot = new int[DataInfo.userNumber + 1];

		for (int i = 0; i < DataInfo.rateNumber; i++)
			userCount[data[i].user]++;

		for (int i = 1; i <= DataInfo.userNumber; i++)
			userPot[i] = userPot[i - 1] + userCount[i - 1];

	}// Of setPot

	/**
	 * 
	 * @param paraUser
	 * @param paraItem
	 * @return null
	 */
	static Triple getDataInfo(int paraUser, int paraItem) {
		int left = userPot[paraUser];
		int right = userPot[paraUser + 1] - 1;

		while (left <= right) {
			int mid = (left + right) / 2;
			if (data[mid].item > paraItem) {
				right = mid - 1;
			} else if (data[mid].item < paraItem) {
				left = mid + 1;
			} else {
				return data[mid];
			} // of if
		} // of while
		return null;
	}// Of getDataInfo

	public static void main(String[] args) {
		try {
 	
			DataInfo tempData = new DataInfo(dataPath);
			SimpleTool.printTriple(tempData.data);
//			SimpleTool.printDoubleArray(uTrAverRates);
//			SimpleTool.printDoubleArray(iTrAverRates);
//			


		} catch (Exception e) {
			e.printStackTrace();
		} // of try-catch
	}// of main
}// Of class DataInfo
