package neurnet;

public class OutputNode extends Node {
	
	/**output nodes don't have weights to the right, so nextWeights and nextGBs are absent**/
	public OutputNode(int nodeIndex, double[] weights, double bias) {
		super(nodeIndex);
		this.weights = weights;
		this.bias = bias;
		this.inputs = new double[weights.length];
		this.gradientWeights = new double[weights.length];
	}
	
	/**solves for activation then returns activation**/
	public double ExtractActivation() {
		SolveForActivation();
		return a;
	}
	
	/**equation for computing gradientbias of the final layer.
	 * this equation is different than if you were to compute the
	 * gradient bias for the hidden layer.**/
	public double ExtractGradientBias(double yvals) {
		gradientBias = (a-yvals) * a * (1-a);
		return gradientBias;
	}
	
	/**equation for extracting the cost of only this node.
	 * cost nodes are the only nodes that influence the cost value**/
	public double ExtractCost(double yvals) {
		return Math.pow((a-yvals), 2);
	}
}
