// Keaton Williams
// Assignment 2 Part 2
// 102-67-511
// 10/21/2025
//
// This program creates a sigmoidal neural network with 784 input nodes,
// 15 hidden nodes, and 10 output nodes. MNIST training and testing data
// are extracted from csv files, and each pixel of any given hand-drawn
// digit is fed into an input node. The network is trained to recognize
// and accurately assess hand-drawn digits. In addition, it can load or
// save previously trained networks, run network accuracy assessments,
// show images and labels that the network is currently reviewing, and
// display information regarding misclassified digits.

package neurnet;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Scanner;

/**main function. create a neural network, then run it**/
@SuppressWarnings("unused")
public class Main {
	
	private static Network N; //the network
	private static Scanner inputReader; //the scanner for reading user input
	
	//whether or not a network has already been loaded or trained. certain
	//options are not available until one of those actions has been done
	private static boolean networkLoaded = false;
	
	/**the main function loads the train/test data, creates a random network,
	 * and prompts the user for various options in an infinite loop that can
	 * only be exitted if the user chooses to. **/
	public static void main(String[] args) throws FileNotFoundException {
		
		System.out.println("Creating a nework and loading data, please wait...\n");

		//extract all data from csv files
		System.out.println("Loading testing data...");
		int[][] testData = ExtractDataFromFile("res/mnist_test.csv");
		System.out.println("Loading training data...");
		int[][] trainData = ExtractDataFromFile("res/mnist_train.csv");
		
		//create the network
		CreateNetwork(trainData);
		
		//run program until it is exitted by the user
		while(true) {
	
			//prompt user with program options and gather user input
			inputReader = new Scanner(System.in);
			int userOption = -1;
			boolean correctInput = false;
			while(!correctInput) {
				correctInput = true;
				PrintUserPrompt();
				
				try { userOption = inputReader.nextInt(); }
				catch (Exception e) {
					System.out.println("\n * ERROR: Please input a valid integer.\n\n");
					inputReader.next();
					correctInput = false;
					continue;
				}
				
				//if a network is loaded, accept inputs 0-7
				if(networkLoaded && (userOption < 0 || userOption > 7)) {
					System.out.println("\n * ERROR: Please input a valid integer.\n\n");
					correctInput = false;
				}
				
				//if there is no network loaded, only accept inputs 0-2
				if(!networkLoaded && (userOption < 0 || userOption > 2)) {
					System.out.println("\n * ERROR: Please input a valid integer.\n\n");
					correctInput = false;
				}
			}
			
			//based on user input, perform desired action
			switch(userOption) {
			
				//exit program
				case 0:
					inputReader.close();
					System.out.println("Exiting program.");
					System.exit(0);
					
				//run network
				case 1:
					RunNetwork();
					break;
					
				//load the network from a file
				case 2:
					try { LoadNetwork(); } 
					catch (Exception e) { System.out.println("File may not have been created yet."); }
					break;
					
				//perform network accuracy on training data
				case 3:
					System.out.println("\nGathering training accuracy details (this may take a while)...");
					N.OneIteration(trainData);
					break;
					
				//perform network accuracy on testing data
				case 4:
					System.out.println("\nGathering testing accuracy details...");
					N.OneIteration(testData);
					break;
					
				//display images of each testing case
				case 5:
					boolean onlyWrongAnswers = false;
					N.DisplayImage(testData, onlyWrongAnswers);
					break;
					
				//display images of only incorrectly guessed testing cases
				case 6:
					onlyWrongAnswers = true;
					N.DisplayImage(testData, onlyWrongAnswers);
					break;
					
				//save the network to one of three different files
				case 7:
					SaveNetwork();
					break;
					
				//something went wrong if this code runs
				default:
					inputReader.close();
					System.out.println("Invalid input. Exiting the program. Relaunch to try again.");
					System.exit(0);
			}
		}
	}
	
	/**creates the network when the program is first launched**/
	private static void CreateNetwork(int[][] trainData) {
		System.out.println("Creating network...");
		N = new Network(trainData);
	}
	
	//////////////////////
	//NETWORK OPERATIONS//
	//////////////////////
	
	/**loads a previously saved network into memory**/
	private static void LoadNetwork() {

		//prompt user with program options and gather user input
		int userOption = -1;
		boolean correctInput = false;
		while(!correctInput) {
			correctInput = true;
			
			//user has the option of picking one of 3 different save files
			System.out.println("\nSelect a file [1-3]:");
			System.out.println("Current files are:");
			if(Files.isRegularFile(Path.of("res/network1.txt"))) {
				System.out.println("[1] network1.txt");
			}
			else { System.out.println("[1] EMPTY"); }
			if(Files.isRegularFile(Path.of("res/network2.txt"))) {
				System.out.println("[2] network2.txt");
			}
			else { System.out.println("[2] EMPTY"); }
			if(Files.isRegularFile(Path.of("res/network3.txt"))) {
				System.out.println("[3] network3.txt");
			}
			else { System.out.println("[3] EMPTY"); }
			System.out.println();
			
			try { userOption = inputReader.nextInt(); }
			catch (Exception e) {
				System.out.println("\n * ERROR: Please input a valid integer.\n\n");
				inputReader.next();
				correctInput = false;
				continue;
			}
			if(userOption < 0 || userOption > 3) { //input validation
				System.out.println("\n * ERROR: Please input a valid integer.\n\n");
				correctInput = false;
			}
		}
		
		//determine file name
		String outputNetworkName = "res/network" + userOption + ".txt";
		
		//take the data from the file, put it into the nodes
		LoadDataFromFile(outputNetworkName);
		N.CreateNodes();
		networkLoaded = true;
		
		System.out.println("\nnetwork" + userOption + ".txt successfully loaded.");
	}
	
	/**runs whatever network is currently loaded**/
	private static void RunNetwork() {
		System.out.println("\nBeginning training (this may take a while)...");
		boolean reviseEnabled = true;
		N.RunTraining();
		networkLoaded = true;
	}
	
	/**saves the current network to one of three files**/
	private static void SaveNetwork() {

		//prompt user with program options and gather user input
		int userOption = -1;
		boolean correctInput = false;
		while(!correctInput) {
			correctInput = true;
			
			//user has the option of picking one of 3 different save files
			//user has the option of picking one of 3 different save files
			System.out.println("\nSelect a file [1-3]:");
			System.out.println("Current files are:");
			if(Files.isRegularFile(Path.of("res/network1.txt"))) {
				System.out.println("[1] network1.txt");
			}
			else { System.out.println("[1] EMPTY"); }
			if(Files.isRegularFile(Path.of("res/network2.txt"))) {
				System.out.println("[2] network2.txt");
			}
			else { System.out.println("[2] EMPTY"); }
			if(Files.isRegularFile(Path.of("res/network3.txt"))) {
				System.out.println("[3] network3.txt");
			}
			else { System.out.println("[3] EMPTY"); }
			System.out.println();
			

			
			try { userOption = inputReader.nextInt(); }
			catch (Exception e) {
				System.out.println("\n * ERROR: Please input a valid integer.\n\n");
				inputReader.next();
				correctInput = false;
				continue;
			}
			
			if(userOption < 0 || userOption > 3) {
				System.out.println("\n * ERROR: Please input a valid integer.\n\n");
				correctInput = false;
			}
		}
		
		//create file name
		String outputNetworkName = "res/network" + userOption + ".txt";
		
		//load the network's weights and biases
		double[][] loadedHiddenWeights = N.GetHiddenWeights();
		double[] loadedHiddenBiases = N.GetHiddenBiases();
		double[][] loadedOutputWeights = N.GetOutputWeights();
		double[] loadedOutputBiases = N.GetOutputBiases();

		//convert data to writeable strings
		String[] hw = new String[loadedHiddenWeights.length];
		String hb = "";
		String[] ow = new String[loadedOutputWeights.length];
		String ob = "";
		
		//convert hidden weights into strings, each number separated by a comma
		String tempStr;
		for(int i=0; i<loadedHiddenWeights.length; i++) { //unpack 2d array
			tempStr = "";
			for(int j=0; j<loadedHiddenWeights[i].length-1; j++) { //for each 1d array, convert it to string with comma
				tempStr += loadedHiddenWeights[i][j] + ",";
			}
			tempStr += loadedHiddenWeights[i][loadedHiddenWeights[i].length-1];
			hw[i] = tempStr;
		}
		
		//convert hidden biases to strings
		for(int i=0; i<loadedHiddenBiases.length-1; i++) {
			hb += loadedHiddenBiases[i] + ",";
		}
		hb += loadedHiddenBiases[loadedHiddenBiases.length-1];
		
		//convert output weights to strings
		for(int i=0; i<loadedOutputWeights.length; i++) {
			tempStr = "";
			for(int j=0; j<loadedOutputWeights[i].length-1; j++) {
				tempStr += loadedOutputWeights[i][j] + ",";
			}
			tempStr += loadedOutputWeights[i][loadedOutputWeights[i].length-1];
			ow[i] = tempStr;
		}
		
		//convert output biases to strings
		for(int i=0; i<loadedOutputBiases.length-1; i++) {
			ob += loadedOutputBiases[i] + ",";
		}
		ob += loadedOutputBiases[loadedOutputBiases.length-1];
		
		//create a file (doesn't create one if it already exists)
		File newOutputFile = new File(outputNetworkName);
		try { newOutputFile.createNewFile(); }
		catch (IOException e) { e.printStackTrace(); }
		
		//create a file writer
		FileWriter fw = null;
		try { fw = new FileWriter(outputNetworkName); }
		catch (IOException e) { e.printStackTrace(); }
		
		//write all contents to file
		try {
			fw.write("Hidden Weights:\n");
			for(int i=0; i<hw.length; i++) {
				fw.write(hw[i]+"\n");
			}
			fw.write("Hidden Biases:\n");
			fw.write(hb + "\n");
			fw.write("Output Weights:\n");
			for(int i=0; i<ow.length; i++) {
				fw.write(ow[i]+"\n");
			}
			fw.write("Output Biases:\n");
			fw.write(ob + "\n");
			fw.close();
		}
		catch (IOException e) { e.printStackTrace(); }
		
		//tell user the file was written
		System.out.println("\nNetwork saved to file " + outputNetworkName);
	}
	
	/**reads network weights and biases from a file and puts them into a data structure for manipulation**/
	private static void LoadDataFromFile(String fileName) {
		
		//open the file with a scanner
		Scanner s = OpenFile(fileName);
		
		//contents of each weight/bias. arraylist was chosen for the ability
		//to use add(), since we cant declare an array size until each line of the file is read
		ArrayList<double[]> hwContents = new ArrayList<double[]>();
		ArrayList<double[]> hbContents = new ArrayList<double[]>();
		ArrayList<double[]> owContents = new ArrayList<double[]>();
		ArrayList<double[]> obContents = new ArrayList<double[]>();
		
		//contents currently being held
		ArrayList<double[]> fileContents = new ArrayList<double[]>();

		//read each line one by one and put it into the array list
		while(s.hasNextLine()) {
			String curLine = s.nextLine();//get the line
			if(curLine.equals("Hidden Weights:")) { continue; } //skip first header
			if(curLine.equals("Hidden Biases:")) {
				hwContents = fileContents; //if we encounter second header, finish hidden weights and skip header
				fileContents = new ArrayList<double[]>();
				continue;
			}
			if(curLine.equals("Output Weights:")) {
				hbContents = fileContents; //third header, finish hidden biases and skip header
				fileContents = new ArrayList<double[]>();
				continue;
			}
			if(curLine.equals("Output Biases:")) {
				owContents = fileContents; //fourth header, finish output weights and skip header
				fileContents = new ArrayList<double[]>();
				continue;
			}
			String[] strDoubles = curLine.split(","); //split the string at each comma
			double[] doubles = new double[strDoubles.length]; //convert the array of strings to an array of ints
			for(int i=0; i<strDoubles.length; i++) {
				doubles[i] = Double.parseDouble(strDoubles[i]);
			}
			fileContents.add(doubles);
		}
		
		//file has ended, finish output biases
		obContents = fileContents;

		//convert back to array and plug everything into the network
		//biases were handled like they were 2d arrays, which i guess they
		//technically were, but they had a length of 1. so for the biases,
		//just unpack the "2d" array and get the actual array of biases
		N.SetHiddenWeights(ConvertArrayListToDoubleArray(hwContents));
		N.SetHiddenBiases(ConvertArrayListToDoubleArray(hbContents)[0]);
		N.SetOutputWeights(ConvertArrayListToDoubleArray(owContents));
		N.SetOutputBiases(ConvertArrayListToDoubleArray(obContents)[0]);
	}
	
	/**read an excel spreadsheet file. convert the contents into a 2d array of ints**/
	private static int[][] ExtractDataFromFile(String fileName) {
		
		//open the file with a scanner
		Scanner s = OpenFile(fileName);
		
		//this is where we will keep the contents of the file. this is an arraylist,
		//not an array, so that i can take advantage of add(). I am not able to know
		//the amount of lines in the file until after the loop has completed.
		ArrayList<int[]> fileContents = new ArrayList<int[]>();

		//read each line one by one and put it into the array list
		while(s.hasNextLine()) {
			String curLine = s.nextLine(); //get the line
			String[] strInts = curLine.split(","); //split the string at each comma
			int[] ints = new int[strInts.length]; //convert the array of strings to an array of ints
			for(int i=0; i<strInts.length; i++) {
				ints[i] = Integer.parseInt(strInts[i]);
			}
			fileContents.add(ints);
		}
		
		//convert the array list of int[]s into int[][]
		return ConvertArrayListToIntArray(fileContents);
	}
	
	/////////////////////////////
	//UTILITY & PRINT FUNCTIONS//
	/////////////////////////////
	
	/**prompts the user with options that they may choose for the program to perform**/
	private static void PrintUserPrompt() {
		
		//only 3 of the 8 options are available if a network hasn't been loaded or trained yet
		if(networkLoaded) {
			System.out.println("\nSelect from one of the available options [0-7]:");
			System.out.println(" * Recommended options are 5 and 6.\n");
			System.out.println("When browsing misclassified images, you might understand why");
			System.out.println("the network got the incorrect answer.\n");
			System.out.println("[1] Train the network.");
			System.out.println("[2] Load a pre-trained network.");
			System.out.println("[3] Display network accuracy on TRAINING data.");
			System.out.println("[4] Display network accuracy on TESTING data.");
			System.out.println("[5] Run network on TESTING data showing images and labels.");
			System.out.println("[6] Display the misclassified TESTING images.");
			System.out.println("[7] Save the network state to a file.");
			System.out.println("[0] Exit.\n");
		}
		else {
			System.out.println("\nSelect from one of the available options [0-2]:");
			System.out.println("More options will become available once a network has been loaded or trained.\n");
			System.out.println("*** A PRE-TRAINED NETWORK HAS ALREADY BEEN PROVIDED FOR YOU.");
			System.out.println("Enter 2 to load a network, then enter 1 to select network 1.\n");
			System.out.println("Training a network may take a long time if your system is not very powerful.\n");
			System.out.println("[1] Train the network.");
			System.out.println("[2] Load a pre-trained network.");
			System.out.println("[0] Exit.\n");
		}
	}
	
	/**converts ArrayList<double[]> --> double[][]**/
	private static double[][] ConvertArrayListToDoubleArray(ArrayList<double[]> list) {
		double[][] array = new double[list.size()][];
		for(int i=0; i< list.size(); i++) {
			array[i] = list.get(i);
		}
		return array;
	}
	
	/**converts ArrayList<int[]> --> int[][]**/
	private static int[][] ConvertArrayListToIntArray(ArrayList<int[]> list) {
		int[][] array = new int[list.size()][];
		for(int i=0; i< list.size(); i++) {
			array[i] = list.get(i);
		}
		return array;
	}
	
	/**opens a file with scanner to read its contents**/
	private static Scanner OpenFile(String fileName) {
		File f = new File(fileName);
		Scanner s = null;
		try { s = new Scanner(f); }
		catch (FileNotFoundException e) {
			System.out.println("\nFile \"" + fileName + "\" not found.");
		}
		return s;
	}
}
