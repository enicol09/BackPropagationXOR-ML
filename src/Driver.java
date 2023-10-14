
import java.io.File;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

/**
 * This class acts as a driver-main. It is the entry point of the program. Here
 * are given from the files all the parameters necessary for the operation of
 * the program. The Drives class is responsible for creating our neural network
 * (type Neurons_Network ) , and to call the functions that will run the
 * training and testing of the neural network as well as is responsible for
 * passing the results to the files calculating the error (training & testing)
 * and the success rate (training & testing).
 * 
 * 
 * Drive produces two files
 *
 * 1. successrate.txt => includes the percentage of correct learning outcomes
 * (training phase) and the percentage of correct results during the test
 * (testing phase). 2. errors.txt => includes the training error at the end of
 * each iteration and the testing error at the end of each iteration.
 *
 * @author Elia Nicolaou 1012334 (enicol09)
 * @version 1.0
 * @see Neurons_Network,Layer
 *
 */
public class Driver {

	// utilities
	private static Scanner input;

	// parameters given from the parameters.txt file
	static String F_parameters = "parameters.txt";
	static int Hidden_layer_2_neurons;
	static int Input_neurons;
	static int Output_neurons;
	static float learning_rate;
	static float momentum;
	static int max_iterations;
	static String train_file;
	static String test_file;
	static int Hidden_layer_1_neurons;

	// arrays for the training & testing data
	static Double[][] training_data;
	static Double[][] testing_data;
	static double target;

	/**
	 * This function is used for checking if a file exists, and if exists will open
	 * it
	 * 
	 * @param filename - the name of the file that we want to open.
	 */
	private static void Check(String filename) {
		File file_name = new File(filename);
		try {

			input = new Scanner(file_name);

		} catch (FileNotFoundException ex) {

			System.out.println(" EROOR: File not Found! . \n");
			System.exit(0);
		}
	}

	/**
	 * The main class...
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {

		// Introduction to print at the console..
		System.out.println(
				" ------ Welcome to the problem XOR with back-Propagation ------- \n Every epoch has one loop of training and one loop of testing"
						+ "\n In our situation we have to bear in mind that every epoch has a Training with the same details as Testing");
		System.out.println("                              ----- Let's begin ------");

		System.out.println("\n--- > Reading Parameters file ... ");

		// opening the parameters file
		Check(F_parameters);

		// reading the parameters of the file " parameters.txt "
		input.next();
		if (input.hasNext())
			Hidden_layer_1_neurons = Integer.parseInt(input.next());

		input.next();
		if (input.hasNext())
			Hidden_layer_2_neurons = input.nextInt();

		input.next();
		if (input.hasNext())
			Input_neurons = input.nextInt();

		input.next();
		if (input.hasNext())
			Output_neurons = input.nextInt();

		input.next();
		if (input.hasNext())
			learning_rate = input.nextFloat();

		input.next();
		if (input.hasNext())
			momentum = input.nextFloat();

		input.next();
		if (input.hasNext())
			max_iterations = input.nextInt();

		input.next();
		if (input.hasNext())
			train_file = input.next();

		input.next();
		if (input.hasNext())
			test_file = input.next();

		input.close();

		System.out.println("--- > Parameters file has been read ... Reading Training File");

		// Opening the training file - we want training file in order to get the values
		// to train our neuron-network.
		Check(train_file);

		int cnt_c = 3;
		// the training file consists of 3
		int cnt_l = 0;

		// finding how much lines
		while (input.hasNextLine()) {
			cnt_l++;
			input.nextLine();
		}

		training_data = new Double[cnt_l][cnt_c]; // initialize the size of the training_data

		int j;
		int i = j = 0;
		input.close();

		Check(train_file);

		while (input.hasNext()) {
			if (j == 3) {
				j = 0;
				i++;
			}
			training_data[i][j] = Double.parseDouble(input.next());
			j++;

		}

		System.out.println("\n--- > Training data has been read ");
		System.out.println("\nTraining Data consists of  " + cnt_l + " lines " + cnt_c + " collumns");

		for (i = 0; i < cnt_l; i++) {
			System.out.println();
			for (j = 0; j < cnt_c; j++) {
				System.out.print(training_data[i][j] + " ");
			}
		}

		System.out.println("\n");
		System.out.println("--- > Reading Testing File ... ");
		Check(test_file);

		cnt_l = 0;
		while (input.hasNextLine()) {
			cnt_l++;
			input.nextLine();
		}

		testing_data = new Double[cnt_l][cnt_c];

		j = 0;
		i = 0;

		input.close();

		// Opening testing_file - we want testing file values in order to test our
		// neuron-network.
		Check(test_file);

		System.out.println("\nTesting Data consists of  " + cnt_l + " lines " + cnt_c + " collumns");

		while (input.hasNext()) {
			if (j == 3) {
				j = 0;
				i++;
			}

			testing_data[i][j] = Double.parseDouble(input.next()); // get the values
			j++;
		}

		for (i = 0; i < cnt_l; i++) {
			System.out.println();
			for (j = 0; j < cnt_c; j++) {
				System.out.print(testing_data[i][j] + " ");
			}
		}

		input.close();

		String error_file = "errors.txt";
		String success_file = "successrate.txt";

		try {

			FileWriter success = new FileWriter(success_file); // creating the file for the success_rate results
			FileWriter error = new FileWriter(error_file); // creating the file for the error results.

			// printing headers
			error.write("\n                 ------------- ERROR FILE ------------- \n");
			error.write("------------------------------------------------------------------------------------\n");
			error.write(" Iterations_counter  |      Training Error     |   Testing Error \n");
			error.write("-------------------------------------------------------------\n");

			success.write("\n                 ------------- SUCCESS FILE ------------- \n");
			success.write("------------------------------------------------------------------------------------\n");
			success.write(" Iterations_counter  |  Training Phase |   Testing Phase \n");
			success.write("-------------------------------------------------------------\n");

			// creating the my neuron network
			Neurons_Network network = new Neurons_Network();

			// printing in the console
			System.out.println(
					"\n \n------------------------------------------------------------------------------------------------ ");
			System.out.print(" Overall we have " + Driver.max_iterations + " Epochs that we want to train & test");
			System.out.println(
					"\n------------------------------------------------------------------------------------------------ ");

			// Starting the training and testing - for each iteration ( every epoch ) we
			// make a training and a testing.
			for (i = 0; i < Driver.max_iterations; i++) {
				
				double error_val = 0;
				int percentage = 0;
				
				//write the current iteration in the files
				error.write("     " + i);
				success.write("   " + i);

				//console printing
				System.out.println("                                Epoch  " + (i + 1));
				System.out.println(
						"\n------------------------------------------------------------------------------------------------ ");
				System.out.println(" Training ... ");

				System.out.println(" Input A :   Input B :      Target  ");
				System.out.println("------------------------------------");

				//training is for all the values of the training data.
				for (int k = 0; k < cnt_l; k++) {
					target = training_data[k][2];
					System.out.println("   " + network.my_neurons[0].output + "          "
							+ network.my_neurons[1].output + "         " + target);

					network.Forward_Propagation(training_data[k][0], training_data[k][1]); //calling forward_propagation function
					network.Back_propagation(target); //calling back_propagation function - needs the target.

					System.out.println("    \n Output = " + network.my_neurons[network.total_neurons - 1].output);
					System.out.println("------------------------------------");
					double output = network.my_neurons[network.total_neurons - 1].output;
					
					//finding the error
					error_val += Math.pow((target - output), 2);

					if (target == 1) {
						if (output >= 0.49)
							percentage += 25;
					}
					if (target == 0) {
						if (output < 0.49)
							percentage += 25;
					}

				}

				
				error.write("                   " + "" + (error_val * (0.5)));
				success.write("                   " + "" + percentage);

				error_val = 0;
				percentage = 0;
				
				//Starting testing
				for (int k = 0; k < cnt_l; k++) {
					target = testing_data[k][2];
					network.Forward_Propagation(testing_data[k][0], testing_data[k][1]);
					double output = network.my_neurons[network.total_neurons - 1].output;
					// find the error
					error_val += Math.pow((target - output), 2);

					if (target == 1) {
						if (output > 0.49)
							percentage += 25;
					}
					if (target == 0) {
						if (output < 0.49)
							percentage += 25;
					}

				}

				error.write("                   " + "" + (error_val * (0.5)) + "\n");
				success.write("                   " + "" + percentage + "\n");

			}

			error.close();
			success.close();
			System.out.println("Successfully wrote to the file.");
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}

	}

}