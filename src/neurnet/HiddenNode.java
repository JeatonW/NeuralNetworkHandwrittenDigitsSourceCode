package neurnet;

public class HiddenNode extends Node {
	
	//the weights of this node connecting to the next layer (for computing hidden layer gradient bias)
	private double[] nextWeights;
	
	/**has everything that input nodes and output nodes have**/
	public HiddenNode(int nodeIndex, double[] weights, double[] nextWeights, double bias) {
		super(nodeIndex);
		this.weights = weights;
		this.nextWeights = nextWeights;
		this.bias = bias;
		this.inputs = new double[weights.length];
		this.gradientWeights = new double[weights.length];
		this.nextGBs = new double[nextWeights.length];
	}
	
	/**compute gradientbias of hidden layer; gradientbias = dot(nextWeights,nextGBs) * a * (1-a).
	 * this equation is different than the output node's equation for gradientbias**/
	public double ExtractGradientBias() {
		double dot = DotProductGBs();
		gradientBias = dot * a * (1-a);
		return gradientBias;
	}
	
	/**performs a dot product on the next layer's weights and GBs**/
	private double DotProductGBs() {
		double total = 0;
		//perform dot product
		for(int i=0; i<nextWeights.length; i++) {
			total += nextWeights[i] * nextGBs[i];
		}
		return total;
	}
	
	/**subtract all the differences from all the next weights.
	 * this type of node is the only one that requires knowing what
	 * the weights of the next layer are.**/
	public void ReviseNextWeights(double[] differences) {
		for(int i=0; i<nextWeights.length; i++) {
			nextWeights[i] -= differences[i];
		}
	}
}
