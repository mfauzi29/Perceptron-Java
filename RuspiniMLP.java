import java.io.IOException;

public class RuspiniMLP {
    public static void main(String[] args) throws IOException {
        MultiLayerPerceptron.runSingleDataset(
                args.length > 0 ? args[0] : "data",
                args.length > 1 ? args[1] : "mlp-output/ruspini",
                "ruspini.csv");
    }
}