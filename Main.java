import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/**
 * 
 * 对七分数据的权重进行平滑，生成六份新数据
 * <br/>
 * 需要注意sigmoid的时候权重为0的项会变成0.5<br/>
 * @author weizhiwei <br/>
 * 2018-11-05 21:35
 */
public class Main {
	public static void main(String[] args) {
//		File f1 = new File("C:\\Users\\weizhiwei"
//				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
//				+ "\\作业\\原始数据\\cross_reference15-18\\2015\\edge2015_1.csv");
//		File f2 = new File("C:\\Users\\weizhiwei"
//				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
//				+ "\\作业\\原始数据\\cross_reference15-18\\2015\\edge2015_2.csv");
//		File f1v = new File("C:\\Users\\weizhiwei"
//				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
//				+ "\\作业\\原始数据\\cross_reference15-18\\2015\\vertex2015_1.csv");
//		File f2v = new File("C:\\Users\\weizhiwei"
//				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
//				+ "\\作业\\原始数据\\cross_reference15-18\\2015\\vertex2015_2.csv");
//		solve(f1,f2,f1v,f2v,"tmp1");
		File f1 = new File("C:\\Users\\weizhiwei"
				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
				+ "\\作业\\原始数据\\edgetmp5.csv");
		File f2 = new File("C:\\Users\\weizhiwei"
				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
				+ "\\作业\\原始数据\\cross_reference15-18\\2018\\edge2018_1.csv");
		File f1v = new File("C:\\Users\\weizhiwei"
				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
				+ "\\作业\\原始数据\\vertextmp5.csv");
		File f2v = new File("C:\\Users\\weizhiwei"
				+ "\\Desktop\\软工课件 (1)\\软件工程课件"
				+ "\\作业\\原始数据\\cross_reference15-18\\2018\\vertex2018_1.csv");
		solve(f1,f2,f1v,f2v,"tmp6");
	}
	private static void solve(File f1, File f2, File f1v, File f2v, String fileName) {
		double alpha = 0.5;
		BufferedReader bf1 = null, bf2 = null;
		Integer[] para = generateVertexFile(f1v, f2v, "C:\\Users\\weizhiwei\\" + 
				"Desktop\\软工课件 (1)\\软件工程课件" + 
				"\\作业\\原始数据\\vertex"+fileName+".csv");
		System.out.println("vertex"+fileName+".csv生成完毕");
		int n0 = para[0];
		int n1 = para[1];
		List<data> allString1 = new ArrayList<>();
		List<data> allString2 = new ArrayList<>();
		try {
			bf1 = new BufferedReader(new FileReader(f1));
			bf2 = new BufferedReader(new FileReader(f2));
		}catch(Exception e) {
			e.printStackTrace();
		}
		String line = "";
		String everyLine = "";
		try {
			line = bf1.readLine();//读取头
			while((line = bf1.readLine()) != null) {
				everyLine = line;
				data dd = new data(everyLine);
				//sigmoid 预处理数据
				dd.weight = 1/(1+Math.pow(Math.E, -dd.weight));
				allString1.add(dd);
			}
			System.out.println(String.format("文件\"%s\"读取完毕", f1.getName()));
		}catch(Exception e) {
			e.printStackTrace();
		}
		double miu1 = 0.0, segma1 = 0.0;
		int cnt = 0;
		try {
			line = bf2.readLine();//读取头
			while((line = bf2.readLine()) != null) {
				everyLine = line;
				cnt++;
				data dd = new data(everyLine);
				dd.weight = 1/(1+Math.pow(Math.E, -dd.weight));
				allString2.add(dd);
				miu1 += dd.weight;
			}
			System.out.println(String.format("文件\"%s\"读取完毕", f2.getName()));
		}catch(Exception e) {
			e.printStackTrace();
		}
		miu1+=(n1*n1-cnt)*0.5;//miu1的初始值应该是0.5
		System.out.println("sigmoid预处理完毕");
		miu1/=n1*n1;//平均值算出来了
		for(data d:allString2) {
			segma1 += (d.weight-miu1)*(d.weight-miu1);
		}
		segma1 += (n1*n1-cnt)*(0.5-miu1)*(0.5-miu1);
		segma1/=n1*n1;//方差算出来了
		allString1.sort(null);
		allString2.sort(null);
		//需要计算alpha!
		alpha = getAlpha(n0,miu1,segma1,allString1);
		System.out.println("alpha是："+alpha);
		System.out.println("miu1是："+miu1);
		System.out.println("segma1是："+segma1);
		List<data> ans = new ArrayList<>();
		int i = 0, j = 0;
		while(i<allString1.size() && j<allString2.size()) {
			data d1 = allString1.get(i);
			data d2 = allString2.get(j);
			if(d1.equals(d2)) {
				i++;j++;
				String wei = String.valueOf(alpha*d1.weight+(1-alpha)*d2.weight);
				ans.add(new data(d1.source+","+d1.target+","+wei));
			}else if(d1.source>d2.source) {
				j++;
				d2.weight = d2.weight*(1-alpha) + 0.5*alpha;
				ans.add(d2);
			}else if(d1.source<d2.source||d1.target<d2.target) {
				i++;
				d1.weight = alpha*d1.weight + 0.5*(1-alpha);
				ans.add(d1);
			}else {
				j++;
				d2.weight = d2.weight*(1-alpha) + 0.5*alpha;
				ans.add(d2);
			}
		}
		while(i<allString1.size()) {
			allString1.get(i).weight = allString1.get(i).weight*alpha + 0.5*(1-alpha);
			ans.add(allString1.get(i));
			i++;
		}
		while(j<allString2.size()) {
			allString2.get(j).weight = allString2.get(j).weight*(1-alpha)+0.5*alpha;
			ans.add(allString2.get(j));
			j++;
		}
		createCSVFile(ans, "C:\\Users\\weizhiwei\\" + 
				"Desktop\\软工课件 (1)\\软件工程课件" + 
				"\\作业\\原始数据\\edge"+fileName+".csv");
		System.out.println("edge"+fileName+".csv生成完毕");
	}
	private static Integer[] generateVertexFile(File f1, File f2, String path) {
		// TODO Auto-generated method stub
		ArrayList<String> ans = new ArrayList<>();
		HashSet<Integer> hs = new HashSet<>();
		Integer[] para = new Integer[2];
		para[0] = new Integer(0);
		para[1] = new Integer(0);
		BufferedReader bf = null;
		try {
			bf = new BufferedReader(new FileReader(f1));
			String line = bf.readLine();
			while((line=bf.readLine())!=null) {
				para[0]++;
				int id = Integer.parseInt(line.split(",")[0]);
				if(!hs.contains(id)) {
					ans.add(line);
					hs.add(id);
				}
			}
			System.out.println(String.format("文件\"%s\"读取完毕", f1.getName()));
		}catch(Exception e) {}
		try {
			bf = new BufferedReader(new FileReader(f2));
			String line = bf.readLine();
			while((line=bf.readLine())!=null) {
				para[1]++;
				int id = Integer.parseInt(line.split(",")[0]);
				if(!hs.contains(id)) {
					ans.add(line);
					hs.add(id);
				}
			}
			System.out.println(String.format("文件\"%s\"读取完毕", f2.getName()));
		}catch(Exception e) {}
		//创建文件
		File csvFile = null;
        BufferedWriter csvWriter = null;
        try {
            csvFile = new File(path);
            File parent = csvFile.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            csvFile.createNewFile();

            // GB2312使正确读取分隔符","
            csvWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                    csvFile), "UTF-8"), 1024);
            // 写入文件头部
            StringBuffer sb = new StringBuffer();
            String rowStr = sb.append("\"").append("Id").append("\",")
            .append("Weight").append(",").
            append("Name").append(",").
            append("Language").append(",").toString();
            csvWriter.write(rowStr);
            csvWriter.newLine();
            for (String s : ans) {
            	String[] d = s.split(",");
                sb = new StringBuffer();
                rowStr = sb.append(d[0]).append(",").
                append(d[1]).append(",").
                append(d[2]).append(",").
                append(d[3]).append(",").toString();
                csvWriter.write(rowStr);
                csvWriter.newLine();
            }
            csvWriter.flush();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
		return para;
	}
	private static double getAlpha(int n0, double miu1, double segma1, List<data> w0) {
		// TODO Auto-generated method stub
		double fsum = 0.0;
		for(int i = 0; i < w0.size(); i++) {
			double wij = w0.get(i).weight;
			fsum += (wij-miu1)*(wij-miu1);
		}
		fsum+=(n0*n0-w0.size())*(0.5-miu1)*(0.5-miu1)+n0*n0*segma1;
		return n0*n0*segma1/fsum;
	}
	private static void createCSVFile(List<data> ans, String path) {
		// TODO Auto-generated method stub
		File csvFile = null;
        BufferedWriter csvWriter = null;
        try {
            csvFile = new File(path);
            File parent = csvFile.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            csvFile.createNewFile();

            // GB2312使正确读取分隔符","
            csvWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                    csvFile), "UTF-8"), 1024);
            // 写入文件头部
            StringBuffer sb = new StringBuffer();
            String rowStr = sb.append("Source").append(",").toString();
            rowStr = sb.append("Target").append(",").toString();
            rowStr = sb.append("Weight").append(",").toString();
            csvWriter.write(rowStr);
            csvWriter.newLine();
            for (data d : ans) {
                sb = new StringBuffer();
                rowStr = sb.append(d.source).append(",").toString();
                rowStr = sb.append(d.target).append(",").toString();
                rowStr = sb.append(-Math.log(1.0/d.weight-1)).toString();
                csvWriter.write(rowStr);
                csvWriter.newLine();
            }
            System.out.println("逆向sigmoid完毕");
            csvWriter.flush();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                csvWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
	}
	static class data implements Comparable<data>{
		int source, target;
		double weight;
		data(data d){
			source = d.source;
			target = d.target;
			weight = d.weight;
		}
		data(String str){
			String[] tmp = str.split(",");
			source = Integer.parseInt(tmp[0]);
			target = Integer.parseInt(tmp[1]);
			weight = Double.parseDouble(tmp[2]);
		}
		@Override
		public String toString() {
			// TODO Auto-generated method stub
			return source+","+target+","+weight;
		}
		@Override
		public boolean equals(Object o) {
			// TODO Auto-generated method stub
			data d = (data)o;
			return source==d.source && target==d.target;
		}
		@Override
		public int compareTo(data d) {
			// TODO Auto-generated method stub
			if(source==d.source)
				return target-d.target;
			return source-d.source;
		}		
	}
}
