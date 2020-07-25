# InstaNet
An experiment into scraping instagram to build a facial recognition network trained across everyone's instagram pages that I follow. There were two main challanges: low shot recognition and identifying the face of the owner of a given page. Low shot recognitoin was difficult because normally CNNs require vast amounts of data to train, most people only have a handfull of pictures on their instagram pages. I overcame this by using a Siamese Neural Network, a network that has been pre-trained on millions of faces and identifies the useful features in distinguishing a given face. When a face is fed into the network, the result is a vector of 128 useful features. The second major challenge was identifying the owner of a given page. People post photos of others, group photos, and memes. To determine which faces belonged to the owner of the page, I used a clustering algorithm to find which faces in someone's profile appeared the most frequently. Then these vectors of people's facial features were added to a database I used to later identify them from the wild. 

Here's an example of the image parsing that takes place in the InstaNet process. 

##### Before: 
![Prior](https://raw.githubusercontent.com/jrockw/FaceNet/master/test.jpg)

##### After:
![Post](https://raw.githubusercontent.com/jrockw/FaceNet/master/RESULT.png)
