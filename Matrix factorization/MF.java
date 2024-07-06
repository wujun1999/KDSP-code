package algorithm;

import java.util.Random;
import datamodel.DataInfo;
import tool.LogWrite1;
import tool.LogWrite2;
import tool.LogWrite3;

public class MF { // 创建了一个类，用public class创建的必须和文件名一样，只能创建一个
	public static String dataPath = DataInfo.dataPath; // static：静态的，作者可以通过类名直接访问，如MF.dataPath ，string 字符串类型
	static String split_Sign = new String("	"); // 被static修饰，类变量，类名.属性名直接访问，有一个地方改变了它，其他地方都会改变

	
	public static double alpha = 0.0001;
	public static double lambda = 0.005;

	public static LogWrite1 logWrite = new LogWrite1();

	/**
	 * 初始化特征矩阵
	 */
	static void initFeature() {
		Random rand = new Random();

		for (int i = 0; i < DataInfo.userNumber; i++)
			for (int j = 0; j < DataInfo.featureNumber; j++) {
				DataInfo.uFeature[i][j] = 1 * rand.nextDouble() - 0.5;
			} // of for j

		for (int i = 0; i < DataInfo.itemNumber; i++)
			for (int j = 0; j < DataInfo.featureNumber; j++) {
				DataInfo.iFeature[i][j] = 1 * rand.nextDouble() - 0.5;
			} // Of for j

	}// of initFeature

	/**
	 * @param userId 用户下标 u
	 * @param itemId 项目下标 v
	 * @return 预测评分 featureNumber == k
	 */
	public static double predict(int userId, int itemId) {
		double preRate = 0;
		for (int i = 0; i < DataInfo.featureNumber; i++) {
			preRate += DataInfo.uFeature[userId][i] * DataInfo.iFeature[itemId][i];
		} // of for i
		return preRate;
	}// Of predict 预测的新的矩阵的每个评分

	/**
	 * 梯度下降，更新潜在特征矩阵上
	 */
	public static void update() {
		for (int i = 0; i < DataInfo.rateNumber; i++) {
			if (i % 10 == DataInfo.teIndxRem) {
				continue;
			} // of if

			int tempUserId = (Integer) DataInfo.data[i].user;
			int tempItemId = (Integer) DataInfo.data[i].item;
			double tempRate = (Double) DataInfo.data[i].rate;

			double tempVary = predict(tempUserId, tempItemId) - tempRate;

			for (int j = 0; j < DataInfo.featureNumber; j++) {
				double tmp = tempVary * DataInfo.iFeature[tempItemId][j] + lambda * DataInfo.uFeature[tempUserId][j];
				DataInfo.uFeature[tempUserId][j] = DataInfo.uFeature[tempUserId][j] - alpha * tmp;
			} // of for j

			for (int j = 0; j < DataInfo.featureNumber; j++) {
				double tmp = tempVary * DataInfo.uFeature[tempUserId][j] + lambda * DataInfo.iFeature[tempItemId][j];
				DataInfo.iFeature[tempItemId][j] = DataInfo.iFeature[tempItemId][j] - alpha * tmp;
			} // of for j
		} // of for i

	}// Of update_one

	/**
	 * 计算真实评分与预测评分之间的RMSE 均方根误差，观测值与真值偏差的平方和观测次数n比值的平方根
	 */
	public double rmse() {
		double rmse = 0;
		int tempTestCount = 0;

		for (int i = 0; i < DataInfo.rateNumber; i++) {
			if (i % 10 != DataInfo.teIndxRem) {
				continue;
			} // of if

			int tempUserIndex = DataInfo.data[i].user;
			int tempItemIndex = DataInfo.data[i].item;
			double tempRate = DataInfo.data[i].rate;

			double prediction = predict(tempUserIndex, tempItemIndex);

//			if (prediction < -10)
//				prediction = -10;
//			if (prediction > 10)
//				prediction = 10;
			
			if (prediction < 0)
				prediction = 0;
			if (prediction > 0.885597)
				prediction = 0.885597;

			prediction = tempRate - prediction;
			rmse += prediction * prediction;
			tempTestCount++;
		} // Of for i
		return Math.sqrt(rmse / tempTestCount);
	}// Of rmse

	/**
	 * 计算真实评分与预测评分之间的MAE
	 */
	public double mae() {
		double mae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < DataInfo.rateNumber; i++) {
			if (i % 10 != DataInfo.teIndxRem) {
				
				continue;
			} // of if

			int tempUserIndex = DataInfo.data[i].user;
			int tempItemIndex = DataInfo.data[i].item;
			double tempRate = DataInfo.data[i].rate;

			double prediction = predict(tempUserIndex, tempItemIndex);

//			if (prediction < -10)
//				prediction = -10;
//			if (prediction > 10)
//				prediction = 10;
			
			if (prediction < 0)
				prediction = 0;
			if (prediction > 0.885597)
				prediction = 0.885597;


			prediction = tempRate - prediction;
			mae += Math.abs(prediction); // 返回绝对值
			tempTestCount++;
		} // Of for i
		return (mae / tempTestCount);
	}// Of mae

	/**
	 * 计算真实评分与预测评分Recall
	 * 
	 * @return Recall
	 */
	public double recall() {
		double R = 0, Rp = 0;// R:真实值>阈值，Rp：预测值>阈值
		double joint = 0;// 真实值和预测值都大于阈值
		for (int i = 0; i < DataInfo.rateNumber; i++) {
			if (i % 10 != DataInfo.teIndxRem) {
				continue;
			} // of if

			int tempUserIndex = DataInfo.data[i].user;
			int tempItemIndex = DataInfo.data[i].item;
			double tempRate = DataInfo.data[i].rate;

			double prediction = predict(tempUserIndex, tempItemIndex);

//			if (prediction < -10)
//				prediction = -10; // 范围锁定在-10到10
//			if (prediction > 10)
//				prediction = 10;
			
			if (prediction < 0)
				prediction = 0;
			if (prediction > 0.885597)
				prediction = 0.885597;

			if (tempRate > DataInfo.theta)
				R++;
			if (prediction > DataInfo.theta)
				Rp++;
			if (tempRate > DataInfo.theta && prediction > DataInfo.theta)
				joint++;
		} // Of for i
		return (joint * 1.0) / R;
	}// Of Recall

	/**
	 * 计算真实评分与预测评分Precision
	 * 
	 * @return Precision
	 */
	public double precision() {
		double R = 0, Rp = 0;// R:真实值>阈值，Rp：预测值>阈值
		double joint = 0;// 真实值和预测值都大于阈值
		for (int i = 0; i < DataInfo.rateNumber; i++) {
			if (i % 10 != DataInfo.teIndxRem) {
				continue;
			} // of if

			
			int tempUserIndex = DataInfo.data[i].user;
			int tempItemIndex = DataInfo.data[i].item;
			double tempRate = DataInfo.data[i].rate;

			double prediction = predict(tempUserIndex, tempItemIndex);

//			if (prediction < -10)
//				prediction = -10;
//			if (prediction > 10)
//				prediction = 10;
			
			if (prediction < 0)
				prediction = 0;
			if (prediction > 0.885597)
				prediction = 0.885597;


			if (tempRate > DataInfo.theta)
				R++;
			if (prediction > DataInfo.theta)
				Rp++;
			if (tempRate > DataInfo.theta && prediction > DataInfo.theta)
				joint++;
		} // Of for i
		return (joint * 1.0) / Rp;
	}// Of Precision

	public static void main(String args[]) {
		try {

			double sumMAE = 0;
			double sumRMSE = 0;
			double sumRecall = 0;
			double sumPre = 0;
			double sumF1 = 0;

			for (int i = 0; i < 10; i++) {
				DataInfo tempData = new DataInfo(dataPath);

				MF tempMF = new MF();

				initFeature();

				System.out.println("Begin Training !");
				int j = 0;
	//			double formerMAE = 0.01995;
				double formerMAE = 0.11;
				double tempMAE = 0;
				double tempRMSE = 0;
				double tempRecall = 0;
				double tempPre = 0;
				double tempF1 = 0;
				update();
				tempMAE = tempMF.mae();
				System.out.println(tempMAE);
				while (formerMAE > tempMAE) {
//				while( j < 1000) {
					formerMAE = tempMAE;
					update();
					tempMAE = tempMF.mae();
					tempRMSE = tempMF.rmse();
					tempPre = tempMF.precision();
					tempRecall = tempMF.recall();
					tempF1 = 2 * tempRecall * tempPre / (tempRecall + tempPre);
					j++;
					System.out.println("round:  " + j + ", MAE: " + tempMAE + ", RMSE: " + tempRMSE + "\n");

				} // of while
				
				if (i == 9) {
//					LogWrite1.log("**************DataInfo.uFeature*****************");
//					LogWrite1.log("\n");
					for (int k = 0; k < DataInfo.userNumber; k++) {
						for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
//							System.out.print(DataInfo.uFeature[k][k2]);
//							System.out.print(" ");
							

							LogWrite1.log(String.valueOf(DataInfo.uFeature[k][k2]));
							LogWrite1.log(" ");
						}
//						System.out.println('\n');
						LogWrite1.log("\n");
					}

					double[][] finalIfeature = new double[DataInfo.featureNumber][DataInfo.itemNumber];

//					System.out.println("**************DataInfo.iFeature*****************");
					for (int k = 0; k < DataInfo.itemNumber; k++) {
						for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
							finalIfeature[k2][k] = DataInfo.iFeature[k][k2];
//							System.out.print(DataInfo.iFeature[k][k2]);
//							System.out.print(" ");
						}
//						System.out.println('\n');

					}

//					LogWrite2.log("**************DataInfo.iFeature*****************");
//					LogWrite2.log("\n");
					for (int k = 0; k < DataInfo.featureNumber; k++) {
						for (int k2 = 0; k2 < DataInfo.itemNumber; k2++) {

//							System.out.print(finalIfeature[k][k2]);
//							System.out.print(" ");
							
							LogWrite2.log(String.valueOf(finalIfeature[k][k2]));
							LogWrite2.log(" ");

						}
//						System.out.println('\n');
						LogWrite2.log("\n");
					}

					double[][] finalMatrix = new double[DataInfo.userNumber][DataInfo.itemNumber];
					for (int k = 0; k < DataInfo.userNumber; k++) {
						for (int k1 = 0; k1 < DataInfo.itemNumber; k1++) {
							for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
								finalMatrix[k][k1] += DataInfo.uFeature[k][k2] * finalIfeature[k2][k1];
							}
						}
					}

//					LogWrite3.log("**************finalMatrix*****************");
//					LogWrite3.log("\n");
					
//					for (int k = 0; k < DataInfo.userNumber; k++) {
//						for (int k2 = 0; k2 < DataInfo.itemNumber; k2++) {
////							System.out.print(finalMatrix[k][k2]);
////							System.out.print(" ");
//							
//
//							LogWrite3.log(String.valueOf(finalMatrix[k][k2]));
//							LogWrite3.log(" ");
//						}
////						System.out.println('\n');
//						LogWrite3.log("\n");
//					}
//					LogWrite3.log("\n");
				}
				
//				LogWrite.log("**************DataInfo.uFeature*****************");
//				for (int k = 0; k < DataInfo.userNumber; k++) {
//					for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
////						System.out.print(DataInfo.uFeature[k][k2]);
////						System.out.print(" ");
//						
//
//						LogWrite.log(String.valueOf(DataInfo.uFeature[k][k2]));
//						LogWrite.log(" ");
//					}
////					System.out.println('\n');
//					LogWrite.log("\n");
//				}
//
//				double[][] finalIfeature = new double[DataInfo.featureNumber][DataInfo.itemNumber];
//
////				System.out.println("**************DataInfo.iFeature*****************");
//				for (int k = 0; k < DataInfo.itemNumber; k++) {
//					for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
//						finalIfeature[k2][k] = DataInfo.iFeature[k][k2];
////						System.out.print(DataInfo.iFeature[k][k2]);
////						System.out.print(" ");
//					}
////					System.out.println('\n');
//
//				}
//
//				LogWrite.log("**************DataInfo.iFeature*****************");
//				for (int k = 0; k < DataInfo.featureNumber; k++) {
//					for (int k2 = 0; k2 < DataInfo.itemNumber; k2++) {
//
////						System.out.print(finalIfeature[k][k2]);
////						System.out.print(" ");
//						
//						LogWrite.log(String.valueOf(finalIfeature[k][k2]));
//						LogWrite.log(" ");
//
//					}
////					System.out.println('\n');
//					LogWrite.log("\n");
//				}
//
//				double[][] finalMatrix = new double[DataInfo.userNumber][DataInfo.itemNumber];
//				for (int k = 0; k < DataInfo.userNumber; k++) {
//					for (int k1 = 0; k1 < DataInfo.itemNumber; k1++) {
//						for (int k2 = 0; k2 < DataInfo.featureNumber; k2++) {
//							finalMatrix[k][k1] += DataInfo.uFeature[k][k2] * finalIfeature[k2][k1];
//						}
//					}
//				}
//
//				LogWrite.log("**************finalMatrix*****************");
//				for (int k = 0; k < DataInfo.userNumber; k++) {
//					for (int k2 = 0; k2 < DataInfo.itemNumber; k2++) {
////						System.out.print(finalMatrix[k][k2]);
////						System.out.print(" ");
//						
//
//						LogWrite.log(String.valueOf(finalMatrix[k][k2]));
//						LogWrite.log(" ");
//					}
////					System.out.println('\n');
//					LogWrite.log("\n");
//				}
//				LogWrite.log("\n");

				if (tempMAE == tempMAE) {
					sumMAE += tempMAE;
					sumRMSE += tempRMSE;
					sumRecall += tempRecall;
					sumPre += tempPre;
					sumF1 += tempF1;
					System.out.println("第 " + (i + 1) + "次：");
					System.out.println("round:  " + j + ", MAE: " + tempMAE + ", RMSE: " + tempRMSE + ", Pre: "
							+ tempPre + ", Recall: " + tempRecall + ",F1: " + tempF1);
				} else
					i--;
			} // of for i

			System.out.println("aveMAE: " + (sumMAE / 10.0));
			System.out.println("aveRMSE: " + (sumRMSE / 10.0));
			System.out.println("aveRecall: " + (sumRecall / 10.0));
			System.out.println("avePre: " + (sumPre / 10.0));
			System.out.println("aveF1: " + (sumF1 / 10.0));
		} catch (Exception e) {
			e.printStackTrace();
		} // of try-catch

	}

	private static double[][] transpose(double[][] iFeature) {
		// TODO Auto-generated method stub
		return null;
	}// of main

}// of class MF
