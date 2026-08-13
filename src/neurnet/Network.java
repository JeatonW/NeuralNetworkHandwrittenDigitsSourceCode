package neurnet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class Network {
	
	/////////////////////////
	//NETWORK INSTANTIATION//
	/////////////////////////
	
	//skipping various sections of the printing for debug purposes
	private boolean skipA2P1Print = true; //assignment 2.1 print stuff
	private boolean skipA2P2Print = false; //assignment 2.2 print stuff
	private boolean skipSeparator = true; //epoch/batch separator
	private boolean skip0A1A = true; //layers 0 and 1 activation values

	//network size given in assignment prompt
	private int inputCount = 784;
	private int hiddenCount = 100;
	private int outputCount = 10;
	
	//lists of all nodes within the network
	private InputNode[] inputNodes = new InputNode[inputCount];
	private HiddenNode[] hiddenNodes = new HiddenNode[hiddenCount];
	private OutputNode[] outputNodes = new OutputNode[outputCount];
	private Node[][] allNodes; //above lists are packed into here when network is created
	
	//learning rates and batch sizes
	private double eta = 2;
	private int batchSize = 10;
	private int totalMiniBatches;
	private int totalEpochs = 30;
	
	//the random initial weights and biases of all nodes located in the hidden/output layers
	private double[][] hiddenWeights, outputWeights;
	private double[] hiddenBiases, outputBiases;
	
	//how much weights and biases will change by after each batch (revisions).
	//revision amount = sumGradients*(eta/batchsize)
	private double[][] revHiddenGWs, revOutputGWs;
	private double[] revHiddenGBs, revOutputGBs;
	
	//the total cost of each training case. starts at 0 and sums up
	private double cost = 0;
	
	//all minibatches chosen from the training data
	double[][][] allBatchData;
	
	/**creates all nodes in the network, giving them the appropriate weights and biases**/
	public Network(int[][] trainData) {
		
		System.out.println("Randomizing weights and biases...");
		RandomizeWeightsBiases();
		System.out.println("Creating the network...");
		CreateNodes();
		System.out.println("Converting training data to batch format...\n");
		ConvertDataToBatches(trainData);
		System.out.println("Complete.\n");
		System.out.println("   -=-=-=-=-=-   ");
	}
	
	/**convert the training data to batch format, and shuffle the training data**/
	private void ConvertDataToBatches(int[][] trainData) {
		
		//60000 / 10 = 6000 minibatches per epoch
		totalMiniBatches = trainData.length / batchSize;
		
		allBatchData = new double[trainData.length][][];
		for(int i=0; i<trainData.length; i++) {
		
			//the first value is the expected answer for this batch, convert it into y values
			//fill the array with zeroes except where index=expected 
			double[] yvals = new double[10];
			yvals[trainData[i][0]] = 1;
			
			//cut off the first index of this array (the y value) so we can handle all x values
			int[] xints = Arrays.copyOfRange(trainData[i], 1, trainData[i].length);
			
			//divide all x values by 255, to put them in a range of 0 to 1.0
			double[] xvals = new double[xints.length];
			for(int j=0; j<xints.length; j++) {
				xvals[j] = (double) xints[j] / 255;
			}
			
			//pack the xvals and yvals together and add it to the list of batches
			//put it in the index according to the shuffled array
			double[][] batch = {xvals, yvals};
			allBatchData[i] = batch;
		}
	}
	
	/**creates all nodes in the network, and assigns their weights and biases**/
	public void CreateNodes() {
		
		//create input nodes
		for(int i=0; i<inputCount; i++) {
			inputNodes[i] = new InputNode(i);
		}
		
		//create hidden nodes
		for(int i=0; i<hiddenCount; i++) {
			//determine the weights that connect this node to the next layer (for gradients)
			//this is essentially a column of outputWeights instead of a row
			double[] nextWeights = new double[outputCount];
			for(int j=0; j<outputCount; j++) {
				nextWeights[j] = outputWeights[j][i];
			}
			hiddenNodes[i] = new HiddenNode(i, hiddenWeights[i], nextWeights, hiddenBiases[i]);
		}
		
		//create output nodes
		for(int i=0; i<outputCount; i++) {
			outputNodes[i] = new OutputNode(i, outputWeights[i], outputBiases[i]);
		}
		
		//pack all layers into one 2D array of nodes
		allNodes = new Node[][] {inputNodes, hiddenNodes, outputNodes};
	}
	
	/**randomizes all initial weights and biases**/
	private void RandomizeWeightsBiases() {
		
		Random r = new Random();
		double min = -1.0;
		double max = 1.0;
		
		//randomize hidden weights from -1.0 to 1.0
		hiddenWeights = new double[hiddenCount][inputCount];
		for(int i=0; i<hiddenCount; i++) {
			for(int j=0; j<inputCount; j++) {
				hiddenWeights[i][j] = min + r.nextDouble() * (max-min);
			}
		}
		
		//randomize hidden biases
		hiddenBiases = new double[hiddenCount];
		for(int i=0; i<hiddenCount; i++) {
			hiddenBiases[i] = min + r.nextDouble() * (max-min);
		}
		
		//randomize output weights
		outputWeights = new double[outputCount][hiddenCount];
		for(int i=0; i<outputCount; i++) {
			for(int j=0; j<hiddenCount; j++) {
				outputWeights[i][j] = min + r.nextDouble() * (max-min);
			}
		}
		
		//randomize output biases
		outputBiases = new double[outputCount];
		for(int i=0; i<outputCount; i++) {
			outputBiases[i] = min + r.nextDouble() * (max-min);
		}
	}
	
	///////////
	//GETTERS//
	///////////
	
	public double[][] GetHiddenWeights() { return hiddenWeights; }
	public double[] GetHiddenBiases() { return hiddenBiases; }
	public double[][] GetOutputWeights() { return outputWeights; }
	public double[] GetOutputBiases() { return outputBiases; }
	
	///////////
	//SETTERS//
	///////////

	public void SetHiddenWeights(double[][] hw) { hiddenWeights = hw; }
	public void SetHiddenBiases(double[] hb) { hiddenBiases = hb; }
	public void SetOutputWeights(double[][] ow) { outputWeights = ow; }
	public void SetOutputBiases(double[] ob) { outputBiases = ob; }
	
	////////////////////////////////////////
	//RUNNING AND MANIPULATING THE NETWORK//
	////////////////////////////////////////
	
	//keep track of which epoch, minibatch, and case we are in
	//(for print output and looping the epoch)
	private int minibatch = 0;
	private int trainingCase = 0;
	private int epochCount = 0;
	
	private int[] writtenNumberTotalGuesses;
	private int[] writtenNumberTotalOccurrences;
	
	/**run batches through the network to train it**/
	public void RunTraining() {
		
		//reset counter vars each time a new training session begins
		epochCount = trainingCase = minibatch = 0;
		
		//loop until epoch count is reached
		while(epochCount < totalEpochs) {
			
			//numbers of correctly classified inputs over total number of occurences
			writtenNumberTotalGuesses = new int[outputCount];
			writtenNumberTotalOccurrences = new int[outputCount];
			
			//shuffle batches between each epoch
			Integer[] shuffledIndexes = ShuffleBatchData();
			
			//perform however many minibatches happen during this epoch
			for(int i=0; i<totalMiniBatches; i++) {
				
				//reset the weight and bias revision amounts between each epoch
				ResetRevisionDifferences();
				
				//perform a minibatch
				for(int j=i*batchSize; j<i*batchSize+batchSize; j++) {
					boolean reviseEnabled = true;
					int[] ay = RunCase(allBatchData[shuffledIndexes[j]], reviseEnabled);
					
					//increment correctly classified inputs and number of occurrences
					writtenNumberTotalOccurrences[ay[1]]++;
					if(ay[0] == ay[1]) { writtenNumberTotalGuesses[ay[0]]++; }
				}
				
				//do revisions at the end of an epoch
				ComputeRevisionAmount();
				PerformRevision();
			
			}

			PrintAccuracy(true);
			
			//increment loop cycle and start over
			epochCount++;
			minibatch = trainingCase = 0;
		}
	}
	
	/**perform only one iteration, without revisions or batches. just one run all the way through, and print accuracy afterwards**/
	public void OneIteration(int[][] data) {
		
		//convert data to batches again (could be switching from training to test or vice versa)
		ConvertDataToBatches(data);
		
		//we are still counting guesses and occurrences, so make sure these exist
		writtenNumberTotalGuesses = new int[outputCount];
		writtenNumberTotalOccurrences = new int[outputCount];
		
		//do each piece of data once
		for(int i=0; i<allBatchData.length; i++) {
			boolean reviseEnabled = false;
			int[] ay = RunCase(allBatchData[i], reviseEnabled);
			
			//increment correctly classified inputs and number of occurrences
			writtenNumberTotalOccurrences[ay[1]]++;
			if(ay[0] == ay[1]) { writtenNumberTotalGuesses[ay[0]]++; }
		}
		
		//print final accuracy
		PrintAccuracy(false);
		
	}
	
	/**perform one iteration, but display an image of the number after each case (among other pieces of information**/
	public void DisplayImage(int[][] testData, boolean onlyWrongAnswers) {

		//convert data to batches again (could be switching from training to test)
		ConvertDataToBatches(testData);
		
		//do each piece of data once
		for(int i=0; i<allBatchData.length; i++) {
			
			boolean reviseEnabled = false; //don't revise during case
			int[] ay = RunCase(allBatchData[i], reviseEnabled); //get expected answer and network guess
			
			String assessment = "INCORRECT"; //whether or not the network assessment is correct
			if(ay[0] == ay[1]) { assessment = "CORRECT"; }
			
			//if we only want to print wrong answers, then only print if theres a wrong answer
			if(!onlyWrongAnswers || ay[0] != ay[1]) {
			
				//print stuff
				System.out.println("\n\n\n\n\n\n\n\n\n\n\n\nCorrect classification: " + ay[1] + "        Network Classification: " + ay[0]);
				DisplayNumber(allBatchData[i]);
				System.out.println("The network's assessment of this case was " + assessment + ".\n");
				System.out.println("Enter 0 to return to the menu. Enter anything else to continue to next case.\n");
				
				//prompt user with program options and gather user input
				@SuppressWarnings("resource")
				Scanner inputReader = new Scanner(System.in);
				String input = inputReader.nextLine();
				if (!input.isBlank()) {
				    int userOption = Integer.parseInt(input);
				    if (userOption == 0) {
				        break;
				    }
				}
			}
		}
	}
	
	/**display an ascii number according to the batch case data**/
	public void DisplayNumber(double[][] batchCase) {
		double[] xvals = batchCase[0];
		int width = (int) Math.sqrt(xvals.length);
		
		int x = 0;
		for(int i=0; i<xvals.length; i++) {
			if(xvals[i] < 0.05) {
				System.out.print(" "); //if the x value is low, print a blank spot
			}
			else {
				System.out.print("#"); //if the x value is high, print a loud spot
			}
			x++;
			if(x >= width) {
				System.out.println(); //if the width of the image is reached, go to next line
				x = 0;
			}
		}
	}
	
	/**shuffles an array of indexes the length of the batch data, returns the answer**/
	private Integer[] ShuffleBatchData() {
		
		//create an array of indexes to shuffle
		Integer[] shuffledIndexes = new Integer[allBatchData.length];
		for(int i=0; i<shuffledIndexes.length; i++) {
			shuffledIndexes[i] = i;
		}

		//convert the array to a list so i can use shuffle on it
		List<Integer> tempList = Arrays.asList(shuffledIndexes);
		Collections.shuffle(tempList);
		
		//convert the list back to an array
		for(int i=0; i< tempList.size(); i++) {
			shuffledIndexes[i] = tempList.get(i);
		}

		return shuffledIndexes;
	}
	
	/**a feed-forward training/testing case with optional back propagation**/
	private int[] RunCase(double[][] batch, boolean reviseEnabled) {
		
		PrintEpochCycle();
		
		double[] xvals = batch[0]; //used to begin the training case
		double[] yvals = batch[1]; //used to assess the training case
		
		//figure out which number we're trying to identify according to y vals
		int currentNumber = -1;
		for(int i=0; i<yvals.length; i++) {
			if(yvals[i] == 1) {
				currentNumber = i;
				break;
			}
		}
		
		//forward pass
		//////////////

		//give all input nodes their value from the batch
		for(int i=0; i<inputNodes.length; i++) { inputNodes[i].ReceiveInputValue(xvals[i]); }
		
		//tell the input nodes to send their value to the hidden layer
		for(int i=0; i<inputNodes.length; i++) { inputNodes[i].PassAValue(hiddenNodes); }
		
		//solve activation value for all hidden nodes
		for(int i=0; i<hiddenNodes.length; i++) { hiddenNodes[i].SolveForActivation(); }
		
		//tell the hidden nodes to send their activation value to the output layer
		for(int i=0; i<hiddenNodes.length; i++) { hiddenNodes[i].PassAValue(outputNodes); }
		
		//solve and activation value for all output nodes
		double[] activationValues = new double[outputCount];
		for(int i=0; i<outputNodes.length; i++) { activationValues[i] = outputNodes[i].ExtractActivation(); }
		
		//determine network's guess as to what the number is
		double max = 0;
		int writtenNumberGuess = -1;
		for(int i=0; i<activationValues.length; i++) {
			if(activationValues[i] > max) {
				max = activationValues[i];
				writtenNumberGuess = i;
			}
		}
		
		PrintForwardHiddenPass();
		PrintForwardOutputPass();
		
		//backwards pass
		////////////////
		
		//only do a backwards pass if we are in training mode
		if(reviseEnabled) {
		
			//solve and extract the output layer's gradient biases.
			//add all biases to the sums of gradients for this training case
			for(int i=0; i<outputNodes.length; i++) {
				revOutputGBs[i] += outputNodes[i].ExtractGradientBias(yvals[i]);
			}
			
			//pass the gradient biases of the output layer to the hidden layer
			for(int i=0; i<outputNodes.length; i++) { outputNodes[i].PassGBValue(hiddenNodes); }
			
			//solve and extract gradient weights.
			//add all weights to the sums of gradients for this training case
			for(int i=0; i<outputNodes.length; i++) {
				//an array is extracted, so do the sum for each weight within the array
				double[] curWeights = outputNodes[i].ExtractGradientWeight();
				for(int j=0; j<curWeights.length; j++) {
					revOutputGWs[i][j] += curWeights[j];
				}
			}
			
			//solve and extract costs.
			//add them all up and divide by 2.
			for(int i=0; i<outputNodes.length; i++) {
				cost += outputNodes[i].ExtractCost(yvals[i]);
			}
			cost /= 2;
	
			//solve and extract the hidden layer's gradient biases.
			//add all biases to the sums of gradients for this training case
			for(int i=0; i<hiddenNodes.length; i++) {
				revHiddenGBs[i] += hiddenNodes[i].ExtractGradientBias();
			}
	
			//solve and extract gradient weights.
			//add all weights to the sums of gradients for this training case
			for(int i=0; i<hiddenNodes.length; i++) {
				double[] curWeights = hiddenNodes[i].ExtractGradientWeight();
				for(int j=0; j<curWeights.length; j++) {
					revHiddenGWs[i][j] += curWeights[j];
				}
			}
	
			PrintBackwardsHiddenPass();
			PrintBackwardsOutputPass();
		
		}
		
		//cost of this training case is reset and we go to the next training case
		cost = 0;
		trainingCase++;
		
		//return what answer the network came up with as well as the expected answer
		int[] ay = new int[2];
		ay[0] = writtenNumberGuess;
		ay[1] = currentNumber;
		return ay;
	}
	
	/**for every gradient sum, multiply it by (learningRate/sizeOfTrainingData) to find how much weights and biases should change**/
	private void ComputeRevisionAmount() {
		
		//learningRate/sizeOfTrainingData
		double x = (double) eta / (double) batchSize;
		
		//layer 1 biases
		for(int i=0; i<revHiddenGBs.length; i++) {
			revHiddenGBs[i] *= x;
		}
		
		//layer 1 weights
		for(int i=0; i<revHiddenGWs.length; i++) {
			for(int j=0; j<revHiddenGWs[0].length; j++) {
				revHiddenGWs[i][j] *= x;
			}
		}
		
		//layer 2 biases
		for(int i=0; i<revOutputGBs.length; i++) {
			revOutputGBs[i] *= x;
		}
		
		//layer 2 weights
		for(int i=0; i<revOutputGWs.length; i++) {
			for(int j=0; j<revOutputGWs[0].length; j++) {
				revOutputGWs[i][j] *= x;
			}
		}
	}
	
	/**give revision values to the nodes so they can perform revision**/
	private void PerformRevision() {
		
		//give all hidden nodes the revision values
		for(int i=0; i<hiddenNodes.length; i++) {
			hiddenNodes[i].ReviseBias(revHiddenGBs[i]);
			hiddenNodes[i].ReviseWeights(revHiddenGWs[i]);
			
			//Note to future self: each node generally only knows the weights that come BEFORE it.
			//Hidden nodes need to know the weights that come AFTER it as well in order
			//to calculate their gradient bias. I made a dumb decision to set these
			//"next weights" equal to the INITIAL weights in CreateNetwork(), so when all of the gradients
			//get revised, the "next weights" don't change and everything breaks. I think this is bad code. 
			//It is fine. However, if we ever have to have more than one layer of hidden
			//nodes, I suspect this will become a nightmare to refactor. For another day I guess.
			double[] diffNextWeights = new double[outputCount];
			for(int j=0; j<outputCount; j++) {
				diffNextWeights[j] = revOutputGWs[j][i];
			}
			hiddenNodes[i].ReviseNextWeights(diffNextWeights);
			
		}
		
		//give all output nodes the revision values
		for(int i=0; i<outputNodes.length; i++) {
			outputNodes[i].ReviseBias(revOutputGBs[i]);
			outputNodes[i].ReviseWeights(revOutputGWs[i]);
		}
		
		PrintRevisions();
		
		//revision just happened so minibatch is complete
		minibatch++;
	}
	
	/**reset revision values back to nothing**/
	private void ResetRevisionDifferences() {
		revHiddenGWs = new double[hiddenCount][inputCount];
		revHiddenGBs = new double[hiddenCount];
		revOutputGWs = new double[outputCount][hiddenCount];
		revOutputGBs = new double[outputCount];
	}
	
	///////////////////
	//PRINT FUNCTIONS//
	///////////////////
	
	/**utility function for converting array of floats to string (so i can debug/print faster)**/
	@SuppressWarnings("unused")
	private String dArrayToString(double[] array) {
		String finalString = "";
		for(int i=0; i<array.length; i++) {
			finalString += array[i] + ", ";
		}
		finalString = finalString.substring(0, finalString.length() - 2);
		return finalString;
	}
	@SuppressWarnings("unused")
	private String iArrayToString(int[] array) {
		String finalString = "";
		for(int i=0; i<array.length; i++) {
			finalString += array[i] + ", ";
		}
		finalString = finalString.substring(0, finalString.length() - 2);
		return finalString;
	}
	@SuppressWarnings("unused")
	private void p(String s) { //im tired of typing out System.out.println() every time
		System.out.println(s);
	}
	
	/**mini-batch & epoch separator for part 1**/
	private void PrintEpochCycle() {
		if(skipSeparator) { return; }
		System.out.println("\n\n     ###############################################################");
		System.out.println("     ##   EPOCH: " + (epochCount+1) + "   ##   MINI-BATCH: " + (minibatch+1) + "   ##   TRAINING CASE: " + (trainingCase+1) + "   ##");
		System.out.println("     ###############################################################\n\n\n");
	}
	
	/**accuracy of guesses for part 2**/
	private void PrintAccuracy(boolean epochsEnabled) {
		if(skipA2P2Print) { return; }
		System.out.println();
		if(epochsEnabled) { System.out.println("Epoch: " + (epochCount+1) + "/" + totalEpochs); }
		int allCorrectGuesses = 0;
		int allOccurrences = 0;
		
		//for each digit 0-9, display how many correct guesses over how many occurrences of that digit
		for(int i=0; i<outputCount; i++) {
			System.out.print(i + " = " + writtenNumberTotalGuesses[i] + "/" + writtenNumberTotalOccurrences[i] + " ");
			allCorrectGuesses += writtenNumberTotalGuesses[i];
			allOccurrences += writtenNumberTotalOccurrences[i];
			if(i == 4) { System.out.println(); } //newline after displaying info for digit 5
		}
		
		//calculate and display overall accuracy
		String accuracy = String.format("%.3f", (double) allCorrectGuesses / (double) allOccurrences * 100);
		System.out.println("\nAccuracy = " + allCorrectGuesses + "/" + allOccurrences + " = " + accuracy + "%");
	}
	
	int inputLayer = 0;
	int hiddenLayer = 1;
	int outputLayer = 2;
	
	private void PrintForwardHiddenPass() {
		if(skipA2P1Print) { return; }
		System.out.println("     * FORWARD PASS - HIDDEN LAYER *\n");
		PrintW(hiddenLayer);
		PrintA(inputLayer);
		PrintB(hiddenLayer);
		PrintZ(hiddenLayer);
		PrintA(hiddenLayer);
	}
	
	private void PrintForwardOutputPass() {
		if(skipA2P1Print) { return; }
		System.out.println("     * FORWARD PASS - OUTPUT LAYER *\n");
		PrintW(outputLayer);
		PrintA(hiddenLayer);
		PrintB(outputLayer);
		PrintZ(outputLayer);
		PrintA(outputLayer);
	}
	
	private void PrintBackwardsHiddenPass() {
		if(skipA2P1Print) { return; }
		System.out.println("     * BACKWARDS PASS - HIDDEN LAYER *\n");
		PrintGB(hiddenLayer);
		PrintGW(hiddenLayer);
	}
	
	private void PrintBackwardsOutputPass() {
		if(skipA2P1Print) { return; }
		System.out.println("     * BACKWARDS PASS - OUTPUT LAYER *\n");
		PrintGB(outputLayer);
		PrintGW(outputLayer);
		System.out.print("COST: " + cost + "\n\n");
	}
	
	private void PrintRevisions() {
		if(skipA2P1Print) { return; }
		System.out.println("     ***** REVISED WEIGHTS & BIASES *****\n");
		PrintB(hiddenLayer);
		PrintW(hiddenLayer);
		PrintB(outputLayer);
		PrintW(outputLayer);
	}
	
	/**print statements for (respectively):
	 *  -weights
	 *  -activation values
	 *  -biases
	 *  -z values
	 *  -gradient biases
	 *  -gradient weights
	 */
	
	private void PrintW(int layer) {
		System.out.println("W: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintW(); }
		System.out.println();
	}
	
	private void PrintA(int layer) {
		if(skip0A1A && layer != 2) { return; }
		System.out.println("A: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintA(); }
		System.out.println();
	}
	
	private void PrintB(int layer) {
		System.out.println("B: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintB(); }
		System.out.println();
	}
	
	private void PrintZ(int layer) {
		System.out.println("Z: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintZ(); }
		System.out.println();
	}
	
	private void PrintGB(int layer) {
		System.out.println("GB: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintGB(); }
		System.out.println();
	}
	
	private void PrintGW(int layer) {
		System.out.println("GW: Layer " + layer);
		for(int i=0; i<allNodes[layer].length; i++) { allNodes[layer][i].PrintGW(); }
		System.out.println();
	}
}
