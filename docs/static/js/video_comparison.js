// Based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// Modified by Dor Verbin, October 2021, updated for Safari support

function clamp(val, min, max) {
    return Math.min(Math.max(val, min), max);
  }
  
  function playVids(videoId) {
      const videoMerge = document.getElementById(videoId + "Merge");
      const vid = document.getElementById(videoId);
  
      let position = 0.5;
      const vidWidth = vid.videoWidth / 2;
      const vidHeight = vid.videoHeight;
  
      const mergeContext = videoMerge.getContext("2d");
  
      function trackLocation(e) {
          const bcr = videoMerge.getBoundingClientRect();
          position = ((e.pageX - bcr.x) / bcr.width);
      }
  
      function trackLocationTouch(e) {
          const bcr = videoMerge.getBoundingClientRect();
          position = ((e.touches[0].pageX - bcr.x) / bcr.width);
      }
  
      videoMerge.addEventListener("mousemove", trackLocation, false); 
      videoMerge.addEventListener("touchstart", trackLocationTouch, false);
      videoMerge.addEventListener("touchmove", trackLocationTouch, false);
  
      function drawLoop() {
          mergeContext.clearRect(0, 0, videoMerge.width, videoMerge.height);
  
          const colStart = clamp(vidWidth * position, 0.0, vidWidth);
          const colWidth = clamp(vidWidth - (vidWidth * position), 0.0, vidWidth);
  
          mergeContext.drawImage(vid,
              vidWidth, 0, vidWidth, vidHeight,
              0, 0, vidWidth, vidHeight
          );
          mergeContext.drawImage(vid,
              colStart, 0, colWidth, vidHeight,
              colStart, 0, colWidth, vidHeight
          );
  
          const arrowLength = 0.09 * vidHeight;
          const arrowheadWidth = 0.025 * vidHeight;
          const arrowheadLength = 0.04 * vidHeight;
          const arrowPosY = vidHeight / 10 * 8;
          const arrowWidth = 0.007 * vidHeight;
          const currX = vidWidth * position;
  
          // Draw circle
          mergeContext.beginPath();
          mergeContext.arc(currX, arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
          mergeContext.fillStyle = "#FFD79340";
          mergeContext.fill();
  
          // Draw border line
          mergeContext.beginPath();
          mergeContext.moveTo(vidWidth * position, 0);
          mergeContext.lineTo(vidWidth * position, vidHeight);
          mergeContext.closePath();
          mergeContext.strokeStyle = "#AAAAAA";
          mergeContext.lineWidth = 5;
          mergeContext.stroke();
  
          // Draw arrow
          mergeContext.beginPath();
          mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);
          mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowWidth / 2);
          mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
          mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
          mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
          mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowWidth / 2);
          mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowWidth / 2);
          mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
          mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
          mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
          mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY);
          mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowWidth / 2);
          mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);
          mergeContext.closePath();
          mergeContext.fillStyle = "#AAAAAA";
          mergeContext.fill();
  
          // Draw text overlays
          mergeContext.font = "40px Arial";
          mergeContext.fillStyle = "white";
          mergeContext.fillText("Our Method", 900, 50);
          mergeContext.fillText("3DGS", 10, 50);
  
          requestAnimationFrame(drawLoop);
      }
  
      // Wait for video to start playing before drawing
      vid.addEventListener("playing", () => {
          requestAnimationFrame(drawLoop);
      });
  
      // Safari needs canplay to trigger before play() works properly
      vid.addEventListener("canplay", () => {
          vid.play();
      });
  }
  
  function resizeAndPlay(element) {
      const cv = document.getElementById(element.id + "Merge");
      cv.width = element.videoWidth / 2;
      cv.height = element.videoHeight;
      element.play();
      element.style.height = "0px"; // Hide video without stopping it
      playVids(element.id);
  }
  
  // Ensure metadata is loaded before resizing
  function initVideo(videoId) {
      const videoEl = document.getElementById(videoId);
      videoEl.muted = true; // Needed for Safari autoplay
      videoEl.addEventListener('loadedmetadata', () => {
          resizeAndPlay(videoEl);
      });
  }
  