package kohonen;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Kohonen {

	int x, y, max_iterations;
	double started_learning_rate, learning_rate, started_neighborhood_size, neighborhood_size;
	neuron[][] grid;

	// winner x, y, distance
	double winner[] = new double[3];

	File testFile,trainFile;
	double[][] trainingData, testData;
	double[] test_error, train_error;
	char[] target_test, target_train;

	/** Constructor of the class */
	public Kohonen(int file_rows, int file_col) {
		try {
			read_parameters();
			started_neighborhood_size = neighborhood_size = x / 2;

			// create grid
			grid = new neuron[x][y];
			for (int i = 0; i < x; i++)
				for (int j = 0; j < y; j++)
					grid[i][j] = new neuron(file_col, i, j);

			trainingData = new double[13333][file_col];
			testData = new double[6667][file_col];

			target_train = new char[13333];
			target_test = new char[6667];

			train_error = new double[max_iterations];
			test_error = new double[max_iterations];

			read_training();
			read_test();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	/** Run Kohonen algorithm */
	public void train() {
		for (int i = 0; i < max_iterations; i++) {
			// train
			for (int j = 0; j < trainingData.length; j++) {
				calculate_distances(trainingData[j]);
				// winner distance
				train_error[i] += winner[2];
				update_weights(trainingData[j]);
			}
			train_error[i] = Math.pow(train_error[i], 2) / trainingData.length;

			// test
			for (int j = 0; j < testData.length; j++) {
				calculate_distances(testData[j]);
				// winner distance
				test_error[i] += winner[2];
			}
			test_error[i] = Math.pow(test_error[i], 2) / testData.length;

			update_neighborhood_size(i);
			update_learning_rate(i);
		}
		store_errors(max_iterations);
	}

	/** LVQ training using test set data */
	public void lvq() {
		for (int i = 0; i < testData.length; i++) {
			calculate_distances(testData[i]);			
			int x = (int) winner[0], y = (int) winner[1];
			if (target_test[i] == grid[x][y].letter)
				grid[x][y].adapt_weights(testData[i], learning_rate, 1);
			else
				grid[x][y].adapt_weights(testData[i], learning_rate, -1);
		}
	}

	/** Calculate distances between every input & weight. Returns the winner */
	public void calculate_distances(double[] inputs) {
		winner[0] = winner[1] = 0;
		winner[2] = Double.MAX_VALUE;

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				grid[i][j].calculateDistance(inputs);
				// store new winner x, y, distance
				if (grid[i][j].distance < winner[2]) {
					winner[0] = i;
					winner[1] = j;
					winner[2] = grid[i][j].distance;
				}
			}
		}
	}

	/** Updates the weights of the winner and it's neighbourhood */
	public void update_weights(double[] inputs) {
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++)
				grid[i][j].adapt_weights(inputs, learning_rate, calculate_neighborhood_function_value(i, j));
	}

	/** Calculates neighbourhood function */
	private double calculate_neighborhood_function_value(int rc_x, int rc_y) {
		double t1, t2;
		// euclidean distance current neuron/winner
		t1 = Math.pow((rc_x - winner[0]), 2) + Math.pow((rc_y - winner[1]), 2);
		t2 = 2 * Math.pow(neighborhood_size, 2);
		return Math.exp(-1.0 * t1 / t2);
	}

	/** Updates the map learning rate */
	public void update_learning_rate(int current_iteration) {
		// temp = -(t/T)
		double temp = -1.0 * ((double) current_iteration / (double) max_iterations);
		// n(t) = n(0) * e^(temp)
		learning_rate = started_learning_rate * Math.exp(temp);
	}

	/** Updates the map neighbourhood size */
	public void update_neighborhood_size(int current_iteration) {
		// A = T / ln(started_neighborhood_size)
		double A = max_iterations / Math.log(started_neighborhood_size);
		// temp = -(t/A)
		double temp = -1.0 * ((double) current_iteration / A);
		neighborhood_size = started_neighborhood_size * Math.exp(temp);
	}

	/** Add labels to grid and stores it in clustering.txt */
	public void labeling(String filename) {
		double min_d = Double.MAX_VALUE;
		int min_pos = -1;
		// for each node in the grid
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				// for each data in the test set
				for (int line = 0; line < target_test.length; line++) {
					grid[i][j].calculateDistance(testData[line]);
					// find the line which is closer to the current neuron
					if (line == 0) {
						min_d = grid[i][j].distance;
						min_pos = line;
					} else if (grid[i][j].distance < min_d) {
						min_d = grid[i][j].distance;
						min_pos = line;
					}
				}
				grid[i][j].letter = target_test[min_pos];
			}
		}
		store_map(filename);
	}

	/** Create clustering.txt and store map */
	public void store_map(String filename) {
		BufferedWriter out = null;
		try {
			out = new BufferedWriter(new FileWriter(filename));
			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++)
					out.write(grid[i][j].letter + " ");
				out.write("\n");
			}
			out.close();
		} catch (IOException e) {
			System.err.println(filename + " cannot be created!");
			e.printStackTrace();
			System.exit(-1);
		}
	}

	/** Read test.txt */
	public void read_test() throws IOException {
		Scanner sc = new Scanner(testFile);
		String s;
		int i = 0;
		while (sc.hasNextLine()) {
			s = sc.nextLine();
			String t[] = s.split(",");
			target_test[i] = t[0].charAt(0);
			for (int j = 1; j < t.length; j++) {
				testData[i][j - 1] = Double.parseDouble(t[j]);
			}
			i++;
		}
		sc.close();
	}

	/** Read training.txt */
	public void read_training() throws IOException {
		Scanner sc = new Scanner(trainFile);
		String s;
		int i = 0;
		while (sc.hasNextLine()) {
			s = sc.nextLine();
			String t[] = s.split(",");
			target_train[i] = t[0].charAt(0);
			for (int j = 1; j < t.length; j++) {
				trainingData[i][j - 1] = Double.parseDouble(t[j]);
			}
			i++;
		}
		sc.close();
	}

	/** Create result.txt file and store errors */
	public void store_errors(int last) {
		BufferedWriter out = null;
		try {
			out = new BufferedWriter(new FileWriter("results.txt"));
			out.write("Iteration\t training_error\t\t\t testing_error\n");
			for (int i = 0; i < last; i++)
				out.write((i + 1) + "\t\t\t " + train_error[i] + "\t " + test_error[i] + "\n");
			out.close();
		} catch (IOException e) {
			System.err.println("errors.txt cannot be created!");
			e.printStackTrace();
			System.exit(-1);
		}
	}

	/** Read parameters.txt */
	public void read_parameters() throws IOException {
		File file = new File("parameters.txt");
		Scanner sc = new Scanner(file);
		String s;

		while (sc.hasNextLine()) {
			s = sc.nextLine();
			String t[] = s.split("\t");

			if (t[0].compareTo("gridWidth") == 0)
				x = Integer.parseInt(t[1]);
			else if (t[0].compareTo("gridHeight") == 0)
				y = Integer.parseInt(t[1]);
			else if (t[0].compareTo("learningRate") == 0)
				started_learning_rate = learning_rate = Float.parseFloat(t[1]);
			else if (t[0].compareTo("maxIterations") == 0)
				max_iterations = Integer.parseInt(t[1]);
			else if (t[0].compareTo("trainFile") == 0)
				trainFile = new File(t[1]);
			else if (t[0].compareTo("testFile") == 0)
				testFile = new File(t[1]);
			else {
				sc.close();
				System.exit(0);
				throw new IOException();
			}
		}
		sc.close();
	}
}