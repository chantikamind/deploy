export class FuzzyART {
  constructor(alpha = 0.0001, beta = 1.0, rho = 0.75) {
    this.alpha = alpha;
    this.beta = beta;
    this.rho = rho;
    this.weights = [];
  }

  choiceFunction(input, weight) {
    let minSum = 0, weightSum = 0;
    for (let i = 0; i < input.length; i++) {
      minSum += Math.min(input[i], weight[i]);
      weightSum += weight[i];
    }
    return minSum / (this.alpha + weightSum);
  }

  matchFunction(input, weight) {
    let minSum = 0, inputSum = 0;
    for (let i = 0; i < input.length; i++) {
      minSum += Math.min(input[i], weight[i]);
      inputSum += input[i];
    }
    return minSum / inputSum;
  }

  train(input) {
    if (this.weights.length === 0) {
      this.weights.push([...input]);
      return 0;
    }

    let bestIdx = -1;
    let bestVal = -Infinity;

    for (let i = 0; i < this.weights.length; i++) {
      const T = this.choiceFunction(input, this.weights[i]);
      if (T > bestVal) {
        bestVal = T;
        bestIdx = i;
      }
    }

    if (this.matchFunction(input, this.weights[bestIdx]) >= this.rho) {
      let W = this.weights[bestIdx];
      for (let i = 0; i < W.length; i++)
        W[i] = this.beta * Math.min(input[i], W[i]) + (1 - this.beta) * W[i];
      return bestIdx;
    } else {
      this.weights.push([...input]);
      return this.weights.length - 1;
    }
  }

  predict(input) {
    let bestIdx = -1, bestVal = -Infinity;

    for (let i = 0; i < this.weights.length; i++) {
      const T = this.choiceFunction(input, this.weights[i]);
      if (T > bestVal) {
        bestVal = T;
        bestIdx = i;
      }
    }

    return bestIdx;
  }
}
