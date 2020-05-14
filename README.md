# InstaNet
An experiment into scraping instagram to build a facial recognition network trained across everyone's instagram pages that I follow. There were two main challanges: low shot recognition and identifying the face of the owner of a given page. Low shot recognitoin was difficult because normally CNNs require vast amounts of data to train, most people only have a handfull of pictures on their instagram pages. I overcame this by using a Siamese Neural Network, a network that has been pre-trained on millions of faces and identifies the useful features in identifying a given face. Then, when a face is fed into the network, the result is a vector of 128 useful features. Secondly, identifying the owner of the page was difficult. I ended up using a clustering algorithm to find which faces in someone's profile were most common. Then these vectors of people's facial features were added to a database I used to later identify them based on a face's similarity to the faces in the database. 

Here's an example of the image parsing that takes place in the InstaNet process. 

Before: 
![Prior](https://raw.githubusercontent.com/jrockw/FaceNet/master/test.jpg)
After:
![Post](https://raw.githubusercontent.com/jrockw/FaceNet/master/RESULT.png)
