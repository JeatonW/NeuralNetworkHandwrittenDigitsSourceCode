package neurnet;

public class InputNode extends Node {

	/**input nodes do not have weights and biases, so they are absent**/
	public InputNode(int nodeIndex) {
		super(nodeIndex);
	}
	
	/**receive the input node's designated value from the batch
	 * this function is different than in the parent function because
	 * input nodes receive their values manually through the network class**/
	public void ReceiveInputValue(double a) {
		this.a = a;
	}
}
