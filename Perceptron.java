import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class Perceptron {
	private static final int MAX_EPOCHS = 1000;
	private static final double LEARNING_RATE = 0.1;

	private static class Dataset {
		final double[][] features;
		final String[] labels;

		Dataset(double[][] features, String[] labels) {
			this.features = features;
			this.labels = labels;
		}
	}

	private static class Model {
		final double[] weights;
		final double bias;

		Model(double[] weights, double bias) {
			this.weights = weights;
			this.bias = bias;
		}

		double score(double[] input) {
			double result = bias;
			for (int index = 0; index < weights.length; index++) {
				result += weights[index] * input[index];
			}
			return result;
		}
	}

	public static void main(String[] args) throws IOException {
		Path dataDirectory = Paths.get(args.length == 0 ? "data" : args[0]);
		if (!Files.isDirectory(dataDirectory)) {
			throw new IOException("Folder data tidak ditemukan: " + dataDirectory.toAbsolutePath());
		}

		List<Path> files = new ArrayList<>();
		Files.list(dataDirectory)
				.filter(path -> path.toString().toLowerCase().endsWith(".csv"))
				.forEach(files::add);
		Collections.sort(files, Comparator.comparing(path -> path.getFileName().toString()));

		if (files.isEmpty()) {
			System.out.println("Tidak ada file CSV di " + dataDirectory.toAbsolutePath());
			return;
		}

		for (Path file : files) {
			runDataset(file);
		}
	}

	private static void runDataset(Path file) throws IOException {
		Dataset dataset = readCsv(file);
		double[][] normalized = standardize(dataset.features);
		Set<String> uniqueLabels = new LinkedHashSet<>(Arrays.asList(dataset.labels));
		List<String> labels = new ArrayList<>(uniqueLabels);
		List<Model> models = new ArrayList<>();

		for (String label : labels) {
			models.add(train(normalized, dataset.labels, label));
		}

		int correct = 0;
		for (int row = 0; row < normalized.length; row++) {
			String prediction = predict(normalized[row], labels, models);
			if (prediction.equals(dataset.labels[row])) {
				correct++;
			}
		}

		double accuracy = 100.0 * correct / normalized.length;
		System.out.printf("%-10s data=%-3d fitur=%-3d kelas=%-2d akurasi=%.2f%%%n",
				file.getFileName(), normalized.length, normalized[0].length, labels.size(), accuracy);
	}

	private static Dataset readCsv(Path file) throws IOException {
		List<double[]> featureRows = new ArrayList<>();
		List<String> labelRows = new ArrayList<>();
		List<String> lines = Files.readAllLines(file);
		int expectedColumns = -1;

		for (String line : lines) {
			if (line.trim().isEmpty()) {
				continue;
			}
			String[] columns = line.split(",", -1);
			if (expectedColumns < 0) {
				expectedColumns = columns.length;
			}
			if (columns.length != expectedColumns) {
				throw new IOException("Jumlah kolom tidak konsisten pada " + file + ": " + line);
			}

			double[] features = new double[columns.length - 1];
			try {
				for (int index = 0; index < features.length; index++) {
					features[index] = Double.parseDouble(columns[index].trim());
				}
				featureRows.add(features);
				labelRows.add(columns[columns.length - 1].trim());
			} catch (NumberFormatException header) {
				if (!featureRows.isEmpty()) {
					throw new IOException("Baris bukan angka pada " + file + ": " + line);
				}
			}
		}

		if (featureRows.isEmpty()) {
			throw new IOException("Dataset kosong: " + file);
		}
		return new Dataset(featureRows.toArray(new double[0][]), labelRows.toArray(new String[0]));
	}

	private static double[][] standardize(double[][] data) {
		int rows = data.length;
		int columns = data[0].length;
		double[][] result = new double[rows][columns];
		for (int column = 0; column < columns; column++) {
			double mean = 0.0;
			for (double[] row : data) {
				mean += row[column];
			}
			mean /= rows;

			double variance = 0.0;
			for (double[] row : data) {
				double difference = row[column] - mean;
				variance += difference * difference;
			}
			double deviation = Math.sqrt(variance / rows);
			for (int row = 0; row < rows; row++) {
				result[row][column] = deviation == 0.0 ? 0.0 : (data[row][column] - mean) / deviation;
			}
		}
		return result;
	}

	private static Model train(double[][] features, String[] labels, String positiveLabel) {
		double[] weights = new double[features[0].length];
		double bias = 0.0;
		for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
			int errors = 0;
			for (int row = 0; row < features.length; row++) {
				int expected = labels[row].equals(positiveLabel) ? 1 : -1;
				double score = bias;
				for (int feature = 0; feature < weights.length; feature++) {
					score += weights[feature] * features[row][feature];
				}
				int actual = score >= 0.0 ? 1 : -1;
				if (actual != expected) {
					for (int feature = 0; feature < weights.length; feature++) {
						weights[feature] += LEARNING_RATE * expected * features[row][feature];
					}
					bias += LEARNING_RATE * expected;
					errors++;
				}
			}
			if (errors == 0) {
				break;
			}
		}
		return new Model(weights, bias);
	}

	private static String predict(double[] features, List<String> labels, List<Model> models) {
		int best = 0;
		double bestScore = models.get(0).score(features);
		for (int index = 1; index < models.size(); index++) {
			double score = models.get(index).score(features);
			if (score > bestScore) {
				best = index;
				bestScore = score;
			}
		}
		return labels.get(best);
	}
}
