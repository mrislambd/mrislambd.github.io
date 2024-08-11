#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
rand.gen<-function(n){
  RN<-vector(length = n)
  x<-rep(n)
  y<-rep(n)
  x[1]<-0;
  y[1]<-1;
  RN[1]<-(x[1]/8+y[1]/7)%% 1
  for (i in 1:n) {
    x[i+1]<-(9*x[i]+3)%% 8
    y[i+1]<-(3*y[i]) %% 7
    RN[i+1]<-(x[i+1]/8+y[i+1]/7)%% 1
  }
  return(data.frame(X_values=x,Y_values=y,R_values=RN))
}
rand.gen(4)  
```  
#
#
#| warning: false
rand.gen<-function(n){
  RN<-vector(length = n)
  x<-rep(n)
  y<-rep(n)
  x[1]<-0;
  y[1]<-1;
  RN[1]<-(x[1]/8+y[1]/7)%% 1
  for (i in 1:n) {
    x[i+1]<-(9*x[i]+3)%% 8
    y[i+1]<-(3*y[i]) %% 7
    RN[i+1]<-(x[i+1]/8+y[i+1]/7)%% 1
  }
  return(data.frame(X_values=x,Y_values=y,R_values=RN))
}
unique(rand.gen(2^10)[3])
#
#
#
#
#
#
#
#
#
#
#
#
#| warning: false
#| message: false
#| echo: false
options(repos=c(CRAN="<something sensible near you>"))
install.packages("dplyr")
library(dplyr)
N=20002
A = matrix(0, ncol=3, nrow=N)
seed <- as.double(1)

RANDU <- function() {
  seed <<- ((2^16 + 3) * seed) %% (2^31)
  round(seed/(2^31),6)
}

for (i in 1:N) {
  A[i,]<-c(RANDU(),RANDU(),RANDU())
}
B=as.data.frame(A)
C<-B %>% filter(V2>=0.5, V2<=0.51)

plot(C$V1,C$V3,xlab = "u_i",ylab = "u_(i+3)")
#
#
#
#
#
#
#
#| warning: false
#| message: false
install.packages("rgl")
install.packages("rglwidget")
install.packages("magick")
library(rgl)
library(magick)

# Set up the 3D plot
N = 1002
A = matrix(0, ncol=3, nrow=N)
seed <- as.double(1)

RANDU <- function() {
  seed <<- ((2^16 + 3) * seed) %% (2^31)
  round(seed/(2^31), 6)
}

for (i in 1:N) {
  A[i, ] <- c(RANDU(), RANDU(), RANDU())
}

B = as.data.frame(A)

# Open a new rgl device and set the background color
rgl.open()
bg3d(color = "#f4f4f4")
plot3d(B$V1, B$V2, B$V3, type="s", size=1, lit=TRUE, col = rainbow(1000))

# Create a directory to store the frames
dir.create("frames")

# Loop to create frames
for (i in 1:360) {
  view3d(userMatrix = rotate3d(matrix = diag(4), angle = i * pi/180, x = 0, y = 1, z = 0))
  rgl.snapshot(filename = sprintf("frames/frame_%03d.png", i))
}

rgl.close()

# Combine frames into a GIF
frames <- image_read(list.files("frames", full.names = TRUE))
animation <- image_animate(frames, fps = 10)
image_write(animation, "3d_plot_animation.gif")

# Cleanup: remove the frames directory
unlink("frames", recursive = TRUE)
#
#
#
#
#
#
