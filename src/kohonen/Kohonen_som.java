package kohonen;

public class Kohonen_som {

	public static void main(String[] args) {

		// 20000 lines of data with 16 features
		Kohonen k = new Kohonen(20000, 16);
		k.train();
		k.labeling("clustering.txt");

		k.lvq();
		k.labeling("clusteringLVQ.txt");
	}
}
