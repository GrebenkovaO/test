// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth/2;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    
    if (vid.readyState > 3) {
        vid.play();

        function trackLocation(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.pageX - bcr.x) / bcr.width);
        }
        function trackLocationTouch(e) {
            // Normalize to [0, 1]
            bcr = videoMerge.getBoundingClientRect();
            position = ((e.touches[0].pageX - bcr.x) / bcr.width);
        }

        videoMerge.addEventListener("mousemove",  trackLocation, false); 
        videoMerge.addEventListener("touchstart", trackLocationTouch, false);
        videoMerge.addEventListener("touchmove",  trackLocationTouch, false);


        function drawLoop() {
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth * position).clamp(0.0, vidWidth);
        
            // Draw left side fully (base layer)
            mergeContext.drawImage(vid,
                0, 0, vidWidth, vidHeight,         // left half of video
                0, 0, vidWidth, vidHeight          // draw to left half of canvas
            );

            // Draw reveal part from the right half
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - colStart).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid,
                vidWidth + colStart, 0, colWidth, vidHeight, // from right half of video
                colStart, 0, colWidth, vidHeight             // draw it over on the canvas
            );
            requestAnimationFrame(drawLoop);
            
            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 10 * 8;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(currX, arrowPosY, arrowLength*0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()
            
            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth*position, 0);
            mergeContext.lineTo(vidWidth*position, vidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = "#AAAAAA";
            mergeContext.lineWidth = 5;            
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth/2);
            
            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowWidth/2);
            
            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY - arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX + arrowLength/2 - arrowheadLength/2, arrowPosY + arrowWidth/2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowWidth/2);
            
            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY + arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY  - arrowheadWidth/2);
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY);
            
            mergeContext.lineTo(currX - arrowLength/2 + arrowheadLength/2, arrowPosY - arrowWidth/2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth/2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#AAAAAA";
            mergeContext.fill();

            // Draw text overlay for our method
            //mergeContext.font = "40px Arial";
            //mergeContext.fillStyle = "white";
            //mergeContext.fillText("Our Method", vidWidth - 200, 50);
            //mergeContext.fillText("3DGS", 10, 50);           
            // mergeContext.fillText("Gaussian", 10, 100);           
            // mergeContext.fillText("Splatting", 10, 150);           
            
        }
        requestAnimationFrame(drawLoop);
    } 
}

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};
    
    
function resizeAndPlay(videoElement) {
    const canvasId = videoElement.id + "Merge";
    const canvas = document.getElementById(canvasId);
  
    if (!canvas) {
      console.warn(`Canvas with ID '${canvasId}' not found.`);
      return;
    }
  
    canvas.width = videoElement.videoWidth / 2;
    canvas.height = videoElement.videoHeight;
  
    videoElement.play();
    videoElement.style.height = "0px"; // Hide the video while it plays
  
    playVids(videoElement.id);
  }