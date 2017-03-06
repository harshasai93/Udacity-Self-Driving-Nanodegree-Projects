#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/canny_edges.jpg "Canny Edges"
[image2]: ./examples/masked_edges.jpg "Masked Edges"
[image3]: ./examples/lines.jpg "Extrapolated Lines"
[image4]: ./examples/img.jpg "Weighted Image"
---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1.First, I converted the images to grayscale, then I applied
Gaussian smoothing / blurring to the images. After smoothing I applied canny edge detection 
algorithm to convert the image into edges, 
![Canny Edges][image1]

2.and applied a quadrilateral mask to crop only the
area of interest which is the lane in the front. 
![Masked Edges][image2]

3.After masking applied hough transform to 
refine the edges obtained from canny edge algorithm according to the given parameters.
Inside the hough lines function the refined lines from hough transform are averaged and 
extrapolated such that a single runs on the left and right of the line. 
![Extrapolated Lines][image3]
![Weighted Image][image4]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 
1. Finding the top-most and bottom-most points in the left and right lines.
2. calculated the slopes of the left and right lanes.
3. As the lines in the farthest from the car are dense and always present, used those top-most
   points and respective slopes to extrapolate to the bottom-most points of the lines.  




###2. Identify potential shortcomings with your current pipeline


1. One potential shortcoming would be what would happen when any of the lane lines
are vertical then slope cannot be calculated, so the lines cannot be extrapolated.
It can happen on curvy roads that one of the lane lines can become vertical.  

2. Sometimes there are negative slope lines in right lane and vice-versa..which has been taken
care by detecting the slope values such that they can be only greater than 0.4 or less than -0.4.
But not able to individually point out the defective lines.

###3. Suggest possible improvements to your pipeline

1. A more enhanced masking which is not hard-coded and determined based on the situation.

2. Individually pointing out the defective lines(slope) in left and right lanes to avoid lines joining 
   from left to right line.

