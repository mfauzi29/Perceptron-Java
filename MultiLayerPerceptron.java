import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MultiLayerPerceptron {
    private static final int MAX_EPOCHS = 1000;
    private static final long SEED = 42L;

    private static class Dataset {
        final double[][] features;
        final String[] labels;

        Dataset(double[][] features, String[] labels) {
            this.features = features;
            this.labels = labels;
        }
    }

    private static class Split {
        final double[][] trainX;
        final String[] trainY;
        final double[][] testX;
        final String[] testY;

        Split(double[][] trainX, String[] trainY, double[][] testX, String[] testY) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.testX = testX;
            this.testY = testY;
        }
    }

    private static class Standardizer {
        final double[] mean;
        final double[] deviation;

        Standardizer(double[][] data) {
            mean = new double[data[0].length];
            deviation = new double[data[0].length];
            for (double[] row : data) {
                for (int column = 0; column < row.length; column++) {
                    mean[column] += row[column];
                }
            }
            for (int column = 0; column < mean.length; column++) {
                mean[column] /= data.length;
            }
            for (double[] row : data) {
                for (int column = 0; column < row.length; column++) {
                    double difference = row[column] - mean[column];
                    deviation[column] += difference * difference;
                }
            }
            for (int column = 0; column < deviation.length; column++) {
                deviation[column] = Math.sqrt(deviation[column] / data.length);
            }
        }

        double[][] transform(double[][] data) {
            double[][] result = new double[data.length][data[0].length];
            for (int row = 0; row < data.length; row++) {
                for (int column = 0; column < data[row].length; column++) {
                    result[row][column] = deviation[column] == 0.0
                            ? 0.0 : (data[row][column] - mean[column]) / deviation[column];
                }
            }
            return result;
        }
    }

    private static class MLP {
        private final int inputSize;
        private final int hiddenSize;
        private final int outputSize;
        private final double[][] hiddenWeights;
        private final double[] hiddenBias;
        private final double[][] outputWeights;
        private final double[] outputBias;
        private final Random random;

        MLP(int inputSize, int hiddenSize, int outputSize, long seed) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.random = new Random(seed);
            hiddenWeights = new double[inputSize][hiddenSize];
            hiddenBias = new double[hiddenSize];
            outputWeights = new double[hiddenSize][outputSize];
            outputBias = new double[outputSize];
            initialize(hiddenWeights, inputSize);
            initialize(outputWeights, hiddenSize);
        }

        private void initialize(double[][] weights, int fanIn) {
            double limit = Math.sqrt(6.0 / (fanIn + weights[0].length));
            for (int row = 0; row < weights.length; row++) {
                for (int column = 0; column < weights[row].length; column++) {
                    weights[row][column] = (random.nextDouble() * 2.0 - 1.0) * limit;
                }
            }
        }

        double train(double[][] inputs, int[] targets, double learningRate, int epochs, Path lossFile)
                throws IOException {
            List<String> lossLines = new ArrayList<>();
            lossLines.add("epoch,loss");
            int[] order = new int[inputs.length];
            for (int index = 0; index < order.length; index++) {
                order[index] = index;
            }
            double lastLoss = Double.POSITIVE_INFINITY;
            int staleEpochs = 0;
            for (int epoch = 1; epoch <= epochs; epoch++) {
                shuffle(order);
                double loss = 0.0;
                for (int position = 0; position < order.length; position++) {
                    int row = order[position];
                    loss += trainOne(inputs[row], targets[row], learningRate);
                }
                loss /= inputs.length;
                lossLines.add(epoch + "," + loss);
                if (lastLoss - loss < 1.0e-7) {
                    staleEpochs++;
                } else {
                    staleEpochs = 0;
                }
                lastLoss = loss;
                if (staleEpochs >= 100) {
                    break;
                }
            }
            Files.write(lossFile, lossLines);
            return lastLoss;
        }

        private double trainOne(double[] input, int target, double learningRate) {
            double[] hidden = new double[hiddenSize];
            for (int hiddenNode = 0; hiddenNode < hiddenSize; hiddenNode++) {
                double sum = hiddenBias[hiddenNode];
                for (int feature = 0; feature < inputSize; feature++) {
                    sum += input[feature] * hiddenWeights[feature][hiddenNode];
                }
                hidden[hiddenNode] = Math.tanh(sum);
            }

            double[] output = softmax(hidden);
            double loss = -Math.log(Math.max(output[target], 1.0e-15));
            double[] outputGradient = output.clone();
            outputGradient[target] -= 1.0;
            for (int hiddenNode = 0; hiddenNode < hiddenSize; hiddenNode++) {
                for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                    outputWeights[hiddenNode][outputNode] -= learningRate
                            * hidden[hiddenNode] * outputGradient[outputNode];
                }
            }
            for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                outputBias[outputNode] -= learningRate * outputGradient[outputNode];
            }

            for (int hiddenNode = 0; hiddenNode < hiddenSize; hiddenNode++) {
                double hiddenGradient = 0.0;
                for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                    hiddenGradient += outputGradient[outputNode] * outputWeights[hiddenNode][outputNode];
                }
                hiddenGradient *= 1.0 - hidden[hiddenNode] * hidden[hiddenNode];
                for (int feature = 0; feature < inputSize; feature++) {
                    hiddenWeights[feature][hiddenNode] -= learningRate * input[feature] * hiddenGradient;
                }
                hiddenBias[hiddenNode] -= learningRate * hiddenGradient;
            }
            return loss;
        }

        private double[] softmax(double[] hidden) {
            double[] logits = new double[outputSize];
            double maximum = -Double.MAX_VALUE;
            for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                logits[outputNode] = outputBias[outputNode];
                for (int hiddenNode = 0; hiddenNode < hiddenSize; hiddenNode++) {
                    logits[outputNode] += hidden[hiddenNode] * outputWeights[hiddenNode][outputNode];
                }
                maximum = Math.max(maximum, logits[outputNode]);
            }
            double total = 0.0;
            for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                logits[outputNode] = Math.exp(logits[outputNode] - maximum);
                total += logits[outputNode];
            }
            for (int outputNode = 0; outputNode < outputSize; outputNode++) {
                logits[outputNode] /= total;
            }
            return logits;
        }

        int predict(double[] input) {
            double[] hidden = new double[hiddenSize];
            for (int hiddenNode = 0; hiddenNode < hiddenSize; hiddenNode++) {
                double sum = hiddenBias[hiddenNode];
                for (int feature = 0; feature < inputSize; feature++) {
                    sum += input[feature] * hiddenWeights[feature][hiddenNode];
                }
                hidden[hiddenNode] = Math.tanh(sum);
            }
            double[] probabilities = softmax(hidden);
            int best = 0;
            for (int outputNode = 1; outputNode < probabilities.length; outputNode++) {
                if (probabilities[outputNode] > probabilities[best]) {
                    best = outputNode;
                }
            }
            return best;
        }

        private void shuffle(int[] values) {
            for (int index = values.length - 1; index > 0; index--) {
                int other = random.nextInt(index + 1);
                int temporary = values[index];
                values[index] = values[other];
                values[other] = temporary;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        Path dataDirectory = Paths.get(args.length == 0 ? "data" : args[0]);
        Path outputDirectory = Paths.get(args.length < 2 ? "mlp-output" : args[1]);
        Files.createDirectories(outputDirectory);
        List<Path> files = new ArrayList<>();
        Files.list(dataDirectory).filter(path -> path.toString().toLowerCase().endsWith(".csv"))
            .filter(path -> isRequestedDataset(path.getFileName().toString()))
                .forEach(files::add);
        Collections.sort(files, Comparator.comparing(path -> path.getFileName().toString()));
        for (Path file : files) {
            runDataset(file, outputDirectory);
        }
    }

    private static boolean isRequestedDataset(String fileName) {
        return fileName.equalsIgnoreCase("XOR.csv") || fileName.equalsIgnoreCase("digit.csv")
                || fileName.equalsIgnoreCase("iris.csv") || fileName.equalsIgnoreCase("ruspini.csv");
    }

    public static void runSingleDataset(String dataDirectory, String outputDirectory, String fileName)
            throws IOException {
        Path dataPath = Paths.get(dataDirectory);
        Path outputPath = Paths.get(outputDirectory);
        Files.createDirectories(outputPath);
        runDataset(dataPath.resolve(fileName), outputPath);
    }

    private static void runDataset(Path file, Path outputDirectory) throws IOException {
        Dataset raw = readCsv(file);
        Map<String, Integer> labelIds = new LinkedHashMap<>();
        for (String label : raw.labels) {
            if (!labelIds.containsKey(label)) {
                labelIds.put(label, labelIds.size());
            }
        }
        Split split = split(raw, file.getFileName().toString().equalsIgnoreCase("XOR.csv"));
        Standardizer standardizer = new Standardizer(split.trainX);
        double[][] trainX = standardizer.transform(split.trainX);
        double[][] testX = standardizer.transform(split.testX);
        int[] trainY = encode(split.trainY, labelIds);
        int[] testY = encode(split.testY, labelIds);

        int[] hiddenOptions = raw.features[0].length > 20 ? new int[] { 16, 32 } : new int[] { 4, 8, 16 };
        double[] learningRates = { 0.03, 0.1, 0.3 };
        MLP bestModel = null;
        double bestLoss = Double.POSITIVE_INFINITY;
        int bestHidden = 0;
        double bestLearningRate = 0.0;
        Path bestLossFile = outputDirectory.resolve(file.getFileName().toString().replace(".csv", "-loss.csv"));
        for (int hidden : hiddenOptions) {
            for (double learningRate : learningRates) {
                MLP candidate = new MLP(trainX[0].length, hidden, labelIds.size(), SEED + hidden);
                Path candidateLoss = outputDirectory.resolve("candidate-loss.csv");
                double loss = candidate.train(trainX, trainY, learningRate, MAX_EPOCHS, candidateLoss);
                if (loss < bestLoss) {
                    bestLoss = loss;
                    bestModel = candidate;
                    bestHidden = hidden;
                    bestLearningRate = learningRate;
                    Files.copy(candidateLoss, bestLossFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }

        if (bestModel == null) {
            throw new IllegalStateException("Model MLP tidak berhasil dilatih untuk " + file);
        }
        int correct = 0;
        for (int row = 0; row < testX.length; row++) {
            if (bestModel.predict(testX[row]) == testY[row]) {
                correct++;
            }
        }
        double accuracy = 100.0 * correct / testX.length;
        System.out.printf("%-10s train=%-3d test=%-3d fitur=%-3d kelas=%-2d hidden=%-2d lr=%.2f loss=%.6f akurasi=%.2f%%%n",
                file.getFileName(), trainX.length, testX.length, trainX[0].length, labelIds.size(),
                bestHidden, bestLearningRate, bestLoss, accuracy);
        if (file.getFileName().toString().equalsIgnoreCase("XOR.csv")) {
            for (int row = 0; row < testX.length; row++) {
                int prediction = bestModel.predict(testX[row]);
                System.out.printf("  XOR %s -> prediksi=%s%n", Arrays.toString(split.testX[row]),
                        split.testY[row].equals(labelIds.keySet().toArray()[prediction]) ? "benar" : "salah");
            }
        }
    }

    private static int[] encode(String[] labels, Map<String, Integer> labelIds) {
        int[] result = new int[labels.length];
        for (int index = 0; index < labels.length; index++) {
            result[index] = labelIds.get(labels[index]);
        }
        return result;
    }

    private static Split split(Dataset dataset, boolean useAllForTinyXor) {
        Map<String, List<Integer>> byLabel = new HashMap<>();
        for (int row = 0; row < dataset.labels.length; row++) {
            byLabel.computeIfAbsent(dataset.labels[row], key -> new ArrayList<>()).add(row);
        }
        List<Integer> train = new ArrayList<>();
        List<Integer> test = new ArrayList<>();
        Random random = new Random(SEED);
        for (List<Integer> rows : byLabel.values()) {
            Collections.shuffle(rows, random);
            int testCount = useAllForTinyXor ? rows.size() : Math.max(1, (int) Math.round(rows.size() * 0.2));
            if (useAllForTinyXor) {
                train.addAll(rows);
                test.addAll(rows);
            } else {
                test.addAll(rows.subList(0, testCount));
                train.addAll(rows.subList(testCount, rows.size()));
            }
        }
        return new Split(select(dataset.features, train), select(dataset.labels, train),
                select(dataset.features, test), select(dataset.labels, test));
    }

    private static double[][] select(double[][] data, List<Integer> indexes) {
        double[][] result = new double[indexes.size()][];
        for (int row = 0; row < indexes.size(); row++) {
            result[row] = data[indexes.get(row)].clone();
        }
        return result;
    }

    private static String[] select(String[] data, List<Integer> indexes) {
        String[] result = new String[indexes.size()];
        for (int row = 0; row < indexes.size(); row++) {
            result[row] = data[indexes.get(row)];
        }
        return result;
    }

    private static Dataset readCsv(Path file) throws IOException {
        List<double[]> features = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        int columnsExpected = -1;
        for (String line : Files.readAllLines(file)) {
            if (line.trim().isEmpty()) {
                continue;
            }
            String[] columns = line.split(",", -1);
            if (columnsExpected < 0) {
                columnsExpected = columns.length;
            }
            if (columns.length != columnsExpected) {
                throw new IOException("Jumlah kolom tidak konsisten pada " + file);
            }
            double[] row = new double[columns.length - 1];
            try {
                for (int column = 0; column < row.length; column++) {
                    row[column] = Double.parseDouble(columns[column].trim());
                }
                features.add(row);
                labels.add(columns[columns.length - 1].trim());
            } catch (NumberFormatException header) {
                if (!features.isEmpty()) {
                    throw new IOException("Baris bukan angka pada " + file + ": " + line);
                }
            }
        }
        if (features.isEmpty()) {
            throw new IOException("Dataset kosong: " + file);
        }
        return new Dataset(features.toArray(new double[0][]), labels.toArray(new String[0]));
    }
}