Neural Network 3D Tic Tac Toe player
Tic Tac Toe is quick game with some simple strategies to win. We thought teaching a neural network to play Tic Tac Toe would be a good way to sharpen our programming and machine learning skills. We coded our model and our utility library in Python 3.7 and used Keras with a Tensorflow backend to build and evaluate our graph.

Tic Tac Toe is a solved game, and there exists a strategy for X's that always wins or ties. We hoped that could expand the space of possible strategies by using a 3D version of Tic Tac Toe. In this version, rows can be filled in straight lines between levels, as well as on the same level. This expands the space of possible strategies.

![https://www.cgtrader.com/3d-models/sports/game/3d-tic-tac-toe](https://img1.cgtrader.com/items/487972/592eb5f7f9/3d-tic-tac-toe-3d-model-low-poly-obj-3ds-fbx-dxf-blend-dae.jpg)

This was project that Josh attempted in 2001 in a course on AI and human cognition. This time, we came at the problem with new tools, and 20 years of neural network research on our side.

In preparing for this project we drew heavily from the work of Michiel van de Steeg, Madalina M. Drugan, and Marco Wiering and their paper: Temporal Difference Learning for the Game Tic-Tac-Toe 3D: Applying Structure to Neural Networks

We were inspired by how they calculated their loss, and their custom layers that helped the AI focus on row completion as a special property of the game.

The Players
Our AIPlayer class takes in a doubled board, one with 0s in empty squares and 1s in the X positions and a simily one concatenated to it for O positions as a flattened array of shape (54,1), and returns a single board (27,1) which is X and O position, plus the AI's move. The model would fill the board with values representing how good a move there would be, between -1 and 1. The model chooses the highest value move from that board for each play.

We also created HumanPlayer class which takes use inputs to determine moves, and an algorithmic player that uses a matrix to analyze the board and look for opportunities to make the maximum progress on the maximus rows each turn, which we called SmartPlayer.

Since the move that SmartPlayer makes is the maximum calculated value from a list, we were able to adjust its 'smarts' by creating a random chance it would choose the 2nd best move, or worse. We called this 'iq' and it is passed to the constructor as a number between 0 and 1, the probability that the SmartPlayer will play the best move. Passing a 1 makes it the smartest we could make it, and passing a 0 makes it play random moves each time.

Training
Our first big hurdle was finding a training set. A training set for a neural network has to be a set of inputs, and labels, or outputs for those inputs. The network learns to match the inputs and outputs. But, what is the value of a move in Tic Tac Toe. The feedback for the game is winning or losing, but the AI won't know this until the end of the game, so how does it choose each move?

This is where we drew on Steeg et. al. They proposed that the label for each move is the value of the move after it. Instead of predicting how to win, the network is instead trying to predict how good this move makes the next one. Eventually the game will end, a final move will be scored, and then weights for earlier moves will trickle back. The network first learns to play end game, and then learns how to get to its favorite endings.

This approach removes the need for a training set of data at all. We don't need some list of games resulting in wins and losses for the network to train on, it can learn by playing the game and by creating it's own predictions and labels.

We tried many variations on opponents, as opponents became our surrogate for training sets. These included algorithmic players, dumbed down algorithmic players, and saved versions of the AIPlayer itself. Training on human opponents was not practical within the scope of this project.

Loss Function
Our loss function ended up being an area we tinkered with a lot. The model looks at it's move from 2 moves before and compares it with the estimated value of the current move. The model seeks to accurately predict its own future valuation of the move. However, the proposed move Yhat and the actual move, Y, are both shape (54,1) boards. In our most basic implementation, we only compare the values for the one move, and do not compare any other values in the array.

However, this implementation of the loss function gave the computer very little feedback on its play. Insights on how to win or avoid losing take a long time to percolate backwards through the moves of a game, and very little information is given at the end. Only the expected value of the single final move, and the final value, -1 or 1, were passed to the model.

Learning was very slow, and our model made many illegal moves. We made other adjustments along the way.

Rules of the Game
We had to choose whether to hardwire the rules of the game into the AIPlayer, or ask it to learn them as it goes. We opted for the latter. We assisted this learning by adjusting the true labels for illegal moves to -1 during any given turn in the loss function. With this adjustment, our models were generally able to learn the rules within a few hundred epochs. The rules being, you can't play where other pieces already are.

Winning
Using these tools, we were able to develop a model that aggressively sought row completion, winning, and was able to learn to beat the SmartPlayer by predicting its traps. SmartPlayer is not able to adapt, so the network would beat it consistently once it found the strategy.

However, the model did not attempt to block their opponent from winning, and fixated on one flawed stategy. We realized that our training set, the SmartPlayer, was too small. The model was overfitting to it.

This is where we developed the IQ for the SmartPlayer, to add some randomizing elements to the training set.

Losing
Even our best models only played offense. They made no attempts to block opponents. We decided that winning quickly and losing slowly were better than winning slowly or losing quickly. In order to pass these values to the model, we added a decay term to the Ytrue that would push values toward 0. This had the effect of decreasing the loss value for a later loss or an earlier win, making those preferable to the model.

Despair
More often, our models would fall into a sort of despair, essentially giving up. This was caused by passing so many -1's in the Y labels for illegal moves. It was also caused by the fact that if our player predicts that every possible move is a loser, and ends up losing, its loss drops to zero. It successfully predicted that it would lose, and predicted all illegal moves. Since we were not giving it any feedback on other moves, those be weighted at -1 without affecting the loss.

We used a few different strategies to combat this "despair."

First, we changed up the opponents. We gave SmartPlayer a randomized IQ setting, and generally trained with a .9 setting. This way the SmartPlayer could still surprise the model sometimes, and the same strategy would not always result in exactly the same loss calculations.

Second, we added an exploration argument to the AIPlayer between 0 and 1. This is similar to the IQ for the SmartPlayer, but in reverse. The exploration number represented how often the model would try something new, a random move rather than following its current strategy. We hoped that this would help break it out of inflexible strategies, as well as prevent zeroing out the loss into despair.

Our models' Yhat predictions for every move were still around -0.9. Pessimism was still the best strategy.

Finally, we added a checker that would analyze the board for opportunities to instantly win, or to block opponents. If a board held a chance to win on the current move, we added a 1 to that position in the Ytrue. Similarly if an opponent would win if the player did not either win or block, a -1 would be placed in every position except the blocking position. We hoped to assist the model in learning these key strategic concepts.

TicRows and TacRows
One of the insights we took from Steeg et. al. was a custom, sparsely connected row, that directed the model to pay special attention to rows rather than individual positions. This layer changes the board representation into a representation values for each row. These values are drawn from the values of each position within the row.

By starting the model graph with this custom layer, we directed the model to look at the board row by row and process it that way.

TacRows does the reverse. It looks at the row by row representation and turns it back into a board representation by combining the values for each row that intersects a given position.

These rows allow the network to process the board as a series of intersections of rows, rather than individual positions. It can set up special monitor hidden units to look for blocking or completion opportunities. The number of hidden units processing each row is adjustable as a hyperparameter.

Results
We were able to train some interesting models, but none that would seriously challenge a human player for more than one or two games. One early model discovered a strategy of creating a triangle diagonally through the board, such as a top right corner on the top board, a bottom right corner on the bottom board, and a bottom left corner on the bottom board. This gave the AI 3 ways to complete a row. A human player that tried to respond would be trapped. However, it could be simply beaten by placing 3 pieces in a row. The strategy took too many moves, and the model did not play defense.

Further Research
This project could be extended to a 4x4x4 grid, and many 4x4x4 versions of Tic Tac Toe exist. The 3x3x3 space actually does not provide a very much more interesting game, as the center position is so powerful. 13 rows intersect it, compared to a maximum of 7 for any other row. X's can always win by taking this position, and unlike 2D Tic Tac Toe, O's cannot force the draw.

Another interesting extension would be to provide web access to the training, and have the model train on games against humans. This could be crowd sourced with an app, or a mobile friendly web page, and would provide a more diverse training experience for the model.
