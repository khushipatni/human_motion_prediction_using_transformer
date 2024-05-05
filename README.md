# Human Motion Prediction Using Transformer

<h2>Architectural Choice</h2>
<p>For our project, we propose to utilize Transformers as the architectural choice for predicting human motion. Transformers offer an effective tool for sequence modeling, especially for capturing long-range relationships, which are critical in motion prediction applications. We will use an encoder-decoder architecture similar to that provided by Martinez et al. [<a href="#ref2">2</a>] describes human posture estimate.</p>

<h2>Proposed Task</h2>
<p>Our main objective is to extrapolate human body motion forward in time, in an unsupervised manner, to predict motion for multiple actions. Specifically, we aim to forecast the most likely future 3D poses of a person given their past motion data. We plan to implement and train a Transformer model to achieve this task.</p>

<h2>Modalities and Datasets</h2>
<p>We will be working with pose data, specifically the Human 3.6M dataset, which provides 3D human motion capture data across various actions. This dataset offers a source of labeled human motion sequences, making it suitable for training and evaluating our prediction model. We have access to this dataset and have identified it as a suitable choice for our project.</p>

<h2>Computational Resources</h2>
<p>While the scope of our project is ambitious, we are mindful of the computational resources available to us. Instead of training the model from scratch on a huge dataset, we want to use transfer learning techniques with pretrained backbones and weights. While the Transformers early layers may capture low-level properties shared by various types of sequential data, later layers might learn domain-specific representations from mocap data during fine-tuning.</p>

<h2>Evaluation Metrics</h2>
<p>We want to minimize the prediction error over a short time horizon (e.g., one second), such that the network can create convincing motion in the near future. To assess the effectiveness of our model, we will compare its predictions to ground truth poses using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and other appropriate measures. We will also compare our Transformer-based model to baseline approaches such as LSTM [<a href="#ref1">1</a>] and Seq-to-Seq [<a href="#ref2">2</a>] to determine its effectiveness.</p>

<h2>References</h2>
<ol>
  <li id="ref1">Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. Recurrent network models for human dynamics. In Proceedings of the IEEE international conference on computer vision, pages 4346-4354, 2015.</li>
  <li id="ref2">Julieta Martinez, Michael J Black, and Javier Romero. On human motion prediction using recurrent neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2891-2900, 2017.</li>
</ol>

<hr>

<h2>Quick demo and visualization</h2>
<p>For a quick demo, you can train for a few iterations and visualize the outputs of your model.</p>

<p>To train, run</p>
<code>python translate.py --verbose True --iterations 1000 --action "walking"</code> <br>

<p>To test the model, run</p>
<code>python translate.py --verbose True --iterations 10 --train False --load 100 --action "walking"</code>  <br>

<p>For visualization, you can look at the approach in the visualizer.ipynb.</p>
