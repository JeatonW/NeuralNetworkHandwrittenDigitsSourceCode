package neurnet;

public abstract class Node {
	
	//index values for location of node within network
	private int nodeIndex;
	
	//various values needed to perform dot product
	protected double[] weights;
	protected double[] inputs;
	protected double bias;
	
	//dot product of inputs and weights
	private double z;
	
	//the activation value
	protected double a;
	
	//gradients
	protected double gradientBias;
	protected double[] gradientWeights;
	
	//the gradient biases of the next layer (for computing hidden layer gradient bias)
	protected double[] nextGBs;
	
	/**instantiate node**/
	public Node(int nodeIndex) {
		this.nodeIndex = nodeIndex;
	}
	
	/////////////////////
	//SOLVING EQUATIONS//
	/////////////////////
	
	/**takes inputs, weights, and biases to solve the activation value**/
	public void SolveForActivation() {
		a = Sigmoidal(DotProductWs());
	}
	
	/**performs dot product using weights and inputs, adds bias. returns the computed Z value**/
	private double DotProductWs() {
		z = bias;
		//perform dot product
		for(int i=0; i<weights.length; i++) {
			z += inputs[i] * weights[i];
		}
		return z;
	}
	
	/**performs sigmoidal function on a Z value and returns**/
	private double Sigmoidal(double z) { 
		return 1 / (1 + Math.pow(Math.E, -z));
	}
	
	/** gradientweight = input*gradientbias, do for all weights and return an array of GWs**/
	public double[] ExtractGradientWeight() {
		for(int i=0; i<inputs.length; i++) {
			gradientWeights[i] = inputs[i]*gradientBias;
		}
		return gradientWeights;
	}
	
	////////////////////////////////////
	//PASSING INFORMATION AMONG LAYERS//
	////////////////////////////////////
	
	/**receive an activation value from a node in the previous layer. place it into inputs array**/
	private void ReceiveAValue(int previousNodeIndex, double valueReceived) {
		inputs[previousNodeIndex] = valueReceived;
	}
	
	/**send this node's activation value to all nodes in the next layer**/
	public void PassAValue(Node[] nextLayer) {
		for(int i=0; i<nextLayer.length; i++) {
			nextLayer[i].ReceiveAValue(nodeIndex, a);
		}
	}
	
	/**receive a gradient bias value from a node in the next layer. place it into nextGBs array**/
	private void ReceiveGBValue(int nextNodeIndex, double GBReceived) {
		nextGBs[nextNodeIndex] = GBReceived;
	}
	
	/**send this node's gradient bias value to all nodes in the previous layer**/
	public void PassGBValue(Node[] prevLayer) {
		for(int i=0; i<prevLayer.length; i++) {
			prevLayer[i].ReceiveGBValue(nodeIndex, gradientBias);
		}
	}
	
	///////////////////////////
	//WEIGHT & BIAS REVISIONS//
	///////////////////////////
	
	/**subtract the difference from the current bias**/
	public void ReviseBias(double difference) {
		bias -= difference;
	}
	
	/**subtract all the differences from all the weights**/
	public void ReviseWeights(double[] differences) {
		for(int i=0; i<weights.length; i++) {
			weights[i] -= differences[i];
		}
	}
	
	///////////////////
	//PRINT FUNCTIONS//
	///////////////////
	
	/**utility convert array of floats to string function**/
	private String dArrayToString(double[] array) {
		String finalString = "";
		for(int i=0; i<array.length; i++) {
			finalString += array[i] + ", ";
		}
		finalString = finalString.substring(0, finalString.length() - 2);
		return finalString;
	}
	
	/**print z value of this node**/
	public void PrintZ() {
		System.out.println(z);
	}
	
	/**print activation value of this node**/
	public void PrintA() {
		System.out.println(a);
	}
	
	/**print weights of this node**/
	public void PrintW() {
		System.out.println(dArrayToString(weights));
	}

	/**print inputs of this node**/
	public void PrintI() {
		System.out.println(dArrayToString(inputs));
	}
	
	/**print bias value of this node**/
	public void PrintB() {
		System.out.println(bias);
	}
	
	/**print gradient bias value of this node**/
	public void PrintGB() {
		System.out.println(gradientBias);
	}
	
	/**print gradient bias values of the next layer**/
	public void PrintNextGB() {
		System.out.println(dArrayToString(nextGBs));
	}
	
	/**print all the gradient weights of this node**/
	public void PrintGW() {
		System.out.println(dArrayToString(gradientWeights));
	}
}