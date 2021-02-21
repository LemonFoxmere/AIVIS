[![MIT License][license-shield]][license-url]

<p align="center">
        <img src="https://aivisualized.com/AIVISLogo.svg" style="width:17%; text-align:center"/>
<h3 align="center">A New Window to Your Neural Network </h3>
        <p align="center">
            You actually read me! Good for you!
            <br />
            <a href="https://aivisualized.com"><strong>Try AIVIS Out Here!</strong></a>
            <br />
          </p>
```
------

<details open="open">
<summary>Table of Contents</summary>
<ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

AIVIS stands for Visualized Artificial Intelligence, and it is an online application aimed to get rid of the headaches of creating and prototyping neural networks. It is perfect for developers like you who wants to experiment and play around with different Neural Network structures, or test out their performances. Its configurable Neural Network allows for up to 9 layers, allowing you to create many network models of your choosing with different complexities in each. AIVIS also allows for the you to upload your own custom input and output dataset, so you can prototype and see the performance in real time. Having it's backbone being supported by the latest version of Tensorflow JS API, it ensures fast training speed and accurate results.

------

## Usage

Basic steps needed to train a neural network:

1. Go to the [AIVIS homepage] (https://aivisualized.com/).
2. Click on the `Get Started` button below the title.
3. Upload the Input and Label files by clicking on the `Upload Input` or `Upload Labels` on the right.
   - Supported file type is limited to CSV files as of now.
   - Every column in the CSV should be a class of data. `(Red, Green, Blue)`
   - Every row in the CSV should be a data point. `(48, 68, 93)`
   - The number of columns in the CSV should be consistent.
   - Example Input and Label files can be found in the Github repository as `x.csv` for inputs, and `Y.csv` for outputs.
     - The provided Input and Label data aims to train a network to recognize the best color, black or white, to put over a certain color background.
4. Configure your Neural Network.
   - Make sure that the amount of input neurons (red circles) matches up with your input data size, and the amount of output neurons (green circles) matches up with your label data size.
     - If the input is for example `(R, G, B, A)`, then the input size should be 4, as there are 4 data points within that input.
     - The example data provided has an input size of 3 `(Red, Green, Blue)`, and an output size of 2 `(Black text, White text)`.
   - You can add or remove layers via the provided buttons on the right.
   - You can change the number of perceptrons within each layer by scrolling up or down on said layer.
5. Input in your training parameters, including Epochs and Batch Sizes
   - Epochs is how many times your neural networks will be trained.
   - Batch Size is how much data should be trained at once.
6. Asssuming you did the above steps correctly, the Train button should lit green, and you may press it to begin the training process.
   - The progress bar will extend on the bottom, and will show the progress of training. The number above the bar will show the loss of the network over time.
7. After your network is trained, you may press test to make a prediction on a random dataset.
   - The input data and prediction will be shown above the progress bar.

Congratulations! You have trained a Neural Network on AIVIS! It's really as simple as that!

------

## Contributing

AIVIS is an amazing place to experiment with Machine Learning and Artificial Intelligence for starters. If you would like to help and make it better, contact me via email! Any contribution that you make are **greatly appreciated**.

You can contribute by:

- Forking this repository.
- Creating your own feature branch, committing and pushing to that branch, and creating a pull request.
- Providing feedback and suggestions via email.

Once again, any contribution will be greatly appreciated and will help AIVIS become a useful, powerful, and intuitive tool to everyone.

------

## License

Distributed under the MIT License. See `LICENSE` for more information.

------

## Contact

Got question or suggestions? Contact me!

Email - reallemonorange@gmail.com

Twitter - [@LemonOrangeTW](https://twitter.com/LemonOrangeTW)
