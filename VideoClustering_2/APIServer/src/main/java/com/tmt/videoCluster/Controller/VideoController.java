package com.tmt.videoCluster.Controller;

import com.tmt.videoCluster.Models.VideoInfo;
import com.tmt.videoCluster.Services.MessageManager;
import com.tmt.videoCluster.Services.VideoDataManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
//import java.util.logging.Logger;

@RestController
@RequestMapping("/video")
public class VideoController {
    private final static Logger logger = LoggerFactory.getLogger(VideoController.class.getName());

    @Autowired
    MessageManager messageManager;

    @Autowired
    VideoDataManager videoDataManager;

    @PostMapping("/info")
    public ResponseEntity<VideoInfo> postVideoInfo(@RequestBody VideoInfo videoInfo) {
        try {
            VideoInfo insertedData = videoDataManager.insertVideoInfo(videoInfo);
            messageManager.sendVideoInfo(insertedData);
            videoDataManager.updateVideoSentTime(insertedData);
            return new ResponseEntity<>(insertedData, HttpStatus.OK);
        } catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/{videoId}/{clusterId}/{productId}")
    public ResponseEntity<VideoInfo> putVideoInfo(@PathVariable String videoId, @PathVariable String clusterId,
                                                  @PathVariable String productId){
        try {
            VideoInfo videoInfo = videoDataManager.updateProductId(videoId, clusterId, productId);
            return new ResponseEntity<>(videoInfo, HttpStatus.OK);
        }
        catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/{videoId}/{clusterId}")
    public ResponseEntity<VideoInfo> putClusterInfo(@PathVariable String videoId, @PathVariable String clusterId,
                                                    @RequestBody VideoInfo.ClusterInfo clusterInfo){
        try {
            VideoInfo videoInfo = videoDataManager.updateVideoClusterInfo(videoId, clusterId, clusterInfo);
            return new ResponseEntity<>(videoInfo, HttpStatus.OK);
        }
        catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PutMapping("/clusterInfos")
    public ResponseEntity<VideoInfo> putClusterInfos(@RequestBody VideoInfo videoInfo){
        try {
            VideoInfo updatedVideoInfo = videoDataManager.updateVideoClusterInfos(videoInfo);
            return new ResponseEntity<>(updatedVideoInfo, HttpStatus.OK);
        } catch (Exception e){
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/info/{videoId}")
    public ResponseEntity<VideoInfo> getVideoInfo(@PathVariable String videoId) {
        try {
            VideoInfo videoInfo = videoDataManager.findVideoById(videoId);
            if (videoInfo == null)
                return new ResponseEntity<>(null, HttpStatus.BAD_REQUEST);
            if (videoInfo.isDone()) {
                return new ResponseEntity<>(videoInfo, HttpStatus.OK);
            } else return new ResponseEntity<>(null, HttpStatus.BAD_REQUEST);
        } catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/info/processing")
    public ResponseEntity<List<VideoInfo>> getProcessingVideoInfo() {
        try {
            List<VideoInfo> videoInfos = videoDataManager.findVideoByStatus(VideoInfo.Status.PROCESSING);
            return new ResponseEntity<>(videoInfos, HttpStatus.OK);
        } catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/info/done")
    public ResponseEntity<List<VideoInfo>> getDoneVideoInfo() {
        try {
            List<VideoInfo> videoInfos = videoDataManager.findVideoByStatus(VideoInfo.Status.DONE);
            return new ResponseEntity<>(videoInfos, HttpStatus.OK);
        } catch (Exception e) {
            logger.error(String.format("Error: %s", e));
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
