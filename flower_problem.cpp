/* flower problem in c++ by @menascii

   input:      
      hardcoded known flower data as a 2d array for training
      hardcoded unkown flower data as a 2d array to make a prediction

   output:
      sample output
      .............
      original weight one: 0.5
      original weight two: 0.3
      original bias: 0.5

      final weight one: 6.34951
      final weight two: 2.80365
      final bias: -20.4011

      target: 1.00000000
      prediction: 1.00000000
      cost: 0.00235834
*/

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>

using namespace std;

double get_z(double length, double height, double weight_one, double weight_two, double bias);
double get_random_number();
double get_sigmoid(double feedValue);
double get_prediction (double z);
double get_cost(double predictionValue, double targetValue);
double get_derivative_sigmoid(double feed);
double get_derivative_cost(double predictionValue, double targetValue);
double get_partial_derivative(double dcost_dpred, double dpred_dz, double partialVariable);

int main()
{
  // training parameters
  double z = 0.0;
  double prediction = 0.0;
  double cost = 0.0;
  double dcost_dpred = 0.0;
  double dpred_dz = 0.0;
  double dcost_dw1 = 0.0;
  double dcost_dw2 = 0.0;
  double dcost_db = 0.0;
  
  // random weight data
  double weight_one = 0.0;
  double weight_two = 0.0;
  double bias = 0.0;

  // change learning rate for tuning
  double learning_rate = 0.2;


  // known flower data to train
  int random_index = 0;
  double rf_length = 0, rf_height = 0, rf_type = 0;
  double flower_data[8][3] =
    {
     {4,     1.5,    1},
     {3,     1.5,    1},
     {3.5,   .5,     1},
     {5.5,   1,      1},
     {2,     1,      0},
     {3,     1,      0},
     {2,     .5,     0},
     {1,     1,      0}
    };

  // unkown flower data to predict after training  
  double unknown_flower [3] =  {5.5, 2, 1};

  cout << endl << endl << endl;
  cout << "######################################################" << endl;
  cout << "          flower problem" << endl;
  cout << "               @      : flower type" << endl;
  cout << "              /  \\    : 1st weight, 2nd weight, bias" << endl;
  cout << "             @    @   : length, width" << endl;
  cout << "######################################################" << endl;

  cout << endl << endl;
  
  
  // used for seeding random numbers
  srand(time(0));

  // assign values to weights
  weight_one = get_random_number();
  weight_two = get_random_number();
  bias = get_random_number();

  cout << "original weight one: " << weight_one << endl;
  cout << "original weight two: " << weight_two << endl;
  cout << "original bias: " << bias << endl << endl;
  
  for(int i = 0; i < 50000; i++)
    {
      // get random index in data set
      random_index = ((float)rand() / (float)RAND_MAX) * (8 - (0));

      // get flower parameters
      rf_length = flower_data[random_index][0];                                                      
      rf_height = flower_data[random_index][1];                                                      
      rf_type = flower_data[random_index][2];

      // get prediction with weighted data
      // cost should verge to limit
      // use print statement to test cost output verging
      z = get_z(rf_length, rf_height, weight_one, weight_two, bias);
      prediction = get_prediction(z);
      cost = get_cost(prediction, rf_type);

      // get derivatives
      dcost_dpred = get_derivative_cost(cost, rf_type);
      dpred_dz = get_derivative_sigmoid(z);

      // get partial derivatives
      dcost_dw1 = get_partial_derivative(dcost_dpred, dpred_dz, rf_length);
      dcost_dw2 = get_partial_derivative(dcost_dpred, dpred_dz, rf_height);
      dcost_db =  get_partial_derivative(dcost_dpred, dpred_dz, 1);

      // update weighted values
      weight_one = weight_one - (learning_rate * dcost_dw1);
      weight_two = weight_two - (learning_rate * dcost_dw2);
      bias = bias - (learning_rate * dcost_db);
    }
  // get unknown target flower to predict with trained weights
  z = get_z(unknown_flower[0], unknown_flower[1], weight_one, weight_two, bias);
  prediction = get_prediction(z);

  cout << "final weight one: " << weight_one << endl;
  cout << "final weight two: " << weight_two << endl;
  cout << "final bias: " << bias << endl << endl;
  cout << "target: " << fixed << setprecision(8) << unknown_flower[2] << endl;
  cout << "prediction: " << fixed << setprecision(8) << prediction << endl;
  cout << "cost: " << fixed << setprecision(8) << cost << endl;
  return 0;
}

double get_z(double length, double height, double weight_one, double weight_two, double bias)
{
  double zValue = 0.0;
  zValue = weight_one * length + weight_two * height + bias;
  return zValue;
}

double get_random_number()
{
  double random = 0.0;
  random = (double)(rand() % 6) / 10.0;
  return random;
}

double get_sigmoid(double feedValue)
{
  double predictionValue = 0.0;
  predictionValue = 1 / (1 + exp(-feedValue));
  return predictionValue;
}

double get_prediction (double z)
{
  double sigmoidValue = 0.0;
  sigmoidValue = get_sigmoid(z);
  return sigmoidValue;
}

double get_cost(double predictionValue, double targetValue)
{
  double costValue = 0.0;
  costValue = pow((predictionValue - targetValue), 2);
  return costValue;
}

double get_derivative_sigmoid(double feedValue)
{
  double derivativeValue = 0.0;
  derivativeValue = get_sigmoid(feedValue) * (1 - get_sigmoid(feedValue));
  return derivativeValue;
}

double get_derivative_cost(double predictionValue, double targetValue)
{
  double derivativeValue = 0.0;
  derivativeValue = 2 * (predictionValue - targetValue);
  return derivativeValue;
}

double get_partial_derivative(double dcost_dpred, double dpred_dz, double partialVariable)
{
  double partialDerivativeValue = 0.0;
  partialDerivativeValue = dcost_dpred * dpred_dz * partialVariable;
  return partialDerivativeValue;
}
