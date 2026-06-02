# Notebooks

Example notebooks demonstrating pysignet. The notebooks form a tutorial sequence; each one builds on the previous.

## [Basic Usage](Basic%20Usage.ipynb)

Introduces the core pysignet workflow: defining [named neuron](https://svivek.com/writing/2026-01-28-named-neurons.html) predicates, constructing logical expressions, and compiling them into differentiable computation graphs. Covers `compile_logic` for per-example evaluation and `logic_to_loss` for training, and explains why soft (t-norm) satisfaction and boolean satisfaction can differ even when both give useful gradients. Also demonstrates the class-selector pattern for multiclass classifiers (`Digit(X, 5)`) and multi-variable constraints.

## [MNIST](MNIST.ipynb)

Shows that expressing digit classification as the logical constraint `Digit(X, Y)` and minimising the resulting loss is mathematically equivalent to standard cross-entropy training. Introduces the key insight that `Y` in a multiclass `nn.Module` predicate is a class selector: pysignet runs the model on `X` and picks the output at index `Y`, rather than treating `Y` as a second model input. Uses `consistency_report` to show that constraint satisfaction and accuracy measure the same thing.

## [Semi-Supervised MNIST](Semi-Supervised%20MNIST.ipynb)

Explores when logical constraints are genuinely useful: when labeled data is scarce. Builds the `exactly_one` constraint by composing `Exists` (at-least-one) and `ForAll` (at-most-one) quantifiers, then applies it as an unsupervised penalty on unlabeled data alongside a supervised loss on 50 labeled examples. Shows that the constraint acts as entropy minimization, reducing prediction uncertainty even without labels. Includes learning curves and an entropy comparison to explain the mechanism, and notes that the supervised signal eventually overrules the constraint as training continues.

## [Symmetry Constraints](Symmetry%20Constraints.ipynb)

Demonstrates how to enforce relational constraints on a binary similarity predicate. Expresses symmetry as `ForAll`-`Equivalent` over `Similar(X1, X2) <-> Similar(X2, X1)`, and explains why models with multiple input arguments must be registered as lambdas rather than `nn.Module` instances directly. Runs a multi-trial experiment across training sizes and plots mean accuracy and symmetry consistency with standard error bands, showing that the constraint provides the strongest relative benefit when training data is scarce.

## [MNIST Addition](MNIST%20Addition.ipynb)

Trains a digit classifier using only indirect supervision: pairs of MNIST images and their sum, with no individual digit labels. Expresses the addition constraint as a `ForAll`-`Implies`-`Exists` expression over all possible sums, and includes a helper predicate that returns zero probability for out-of-range indices. After 15 epochs the model reaches ~97.8% digit accuracy, matching a supervised baseline trained with full labels.
