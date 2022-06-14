package com.tmt.videoCluster.Services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.tmt.videoCluster.Models.VideoInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.mongodb.core.query.Update;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.Date;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class VideoDataManager {
    private final static Logger logger = LoggerFactory.getLogger(VideoDataManager.class.getName());
    private final static long oneDayTimeStamp = 300000;

    @Autowired
    MongoOperations mongoOperations;

    @Autowired
    MessageManager messageManager;

    public VideoInfo insertVideoInfo(VideoInfo videoInfo){
        if (mongoOperations.exists(Query.query(Criteria.where("id").is(videoInfo.getId())), VideoInfo.class))
            return videoInfo;
        return mongoOperations.insert(videoInfo);
    }

    public VideoInfo findVideoById(String videoId){
        return mongoOperations.findOne(Query.query(Criteria.where("id").is(videoId)), VideoInfo.class);
    }

    public List<VideoInfo> findVideoByStatus(VideoInfo.Status status){
        return mongoOperations.find(Query.query(Criteria.where("status").is(status))
                .with(Sort.by("doneTime").descending()), VideoInfo.class);
    }

    public void updateVideoStatus(String videoId, VideoInfo.Status videoStatus){
        mongoOperations.updateFirst(Query.query(Criteria.where("id").is(videoId)),
                new Update()
                        .set("status", videoStatus),
                VideoInfo.class);
        logger.info("Update Status of Video " + videoId + " to " + videoStatus);
    }

    public void updateVideoResult(VideoInfo videoInfo){
        long currentTime = new Date().getTime();
        mongoOperations.upsert(Query
                        .query(Criteria.where("id").is(videoInfo.getId())),
                new Update()
                        .setOnInsert("videoUrl", videoInfo.getVideoUrl())
                        .setOnInsert("createdTime", currentTime)
                        .setOnInsert("sentTime", currentTime)
                        .set("status", videoInfo.getStatus())
                        .set("clusterInfos", videoInfo.getClusterInfos())
                        .set("doneTime", currentTime),
                VideoInfo.class);
        logger.info(String.format("Updated video %s", videoInfo.getId()));
    }

    public VideoInfo updateVideoClusterInfos(VideoInfo videoInfo){
        mongoOperations.updateFirst(Query.query(Criteria.where("id").is(videoInfo.getId())),
                new Update().set("clusterInfos", videoInfo.getClusterInfos()),
                VideoInfo.class);
        logger.info(String.format("Manually Updated video %s", videoInfo.getId()));
        return mongoOperations.findOne(Query.query(Criteria.where("id").is(videoInfo.getId())), VideoInfo.class);
    }

    public VideoInfo updateVideoClusterInfo(String videoId, String clusterId, VideoInfo.ClusterInfo clusterInfo){
        clusterInfo.setId(clusterId);
        mongoOperations.updateFirst(Query.query(Criteria
                        .where("id").is(videoId)
                        .and("status").is(VideoInfo.Status.DONE)
                        .and("clusterInfos.id").is(clusterId)),
                new Update()
                        .set("clusterInfos.$", clusterInfo),
                VideoInfo.class);
        return mongoOperations.findOne(Query.query(Criteria.where("id").is(videoId)), VideoInfo.class);
    }

    public VideoInfo updateProductId(String videoId, String clusterId, String productId){
        mongoOperations.updateFirst(Query.query(Criteria
                        .where("id").is(videoId)
                        .and("status").is(VideoInfo.Status.DONE)
                        .and("clusterInfos.id").is(clusterId)),
                new Update()
                        .set("clusterInfos.$.productId", productId),
                VideoInfo.class);
        return mongoOperations.findOne(Query.query(Criteria.where("id").is(videoId)), VideoInfo.class);
    }

    public List<VideoInfo> getSlowItems(){
        long currentTime = new Date().getTime();
        return mongoOperations.find(Query.query(Criteria
                .where("sentTime").lte(currentTime - oneDayTimeStamp)
                .gte(currentTime - oneDayTimeStamp * 2)
                .and("status").ne(VideoInfo.Status.DONE)), VideoInfo.class);
    }

    public void updateVideoSentTime(VideoInfo videoInfo){
        mongoOperations.updateFirst(Query.query(Criteria.where("id").is(videoInfo.getId())),
                new Update().set("sentTime", new Date().getTime()),
                VideoInfo.class);
    }

    private void updateClusteringResultJob(){
        logger.info("Start UpdateClusteringResultJob");
        while (true){
            try{
                VideoInfo videoInfo = messageManager.getClusteringResult();
                if (videoInfo == null) continue;
                updateVideoResult(videoInfo);
                logger.info("Update Video " + videoInfo.getId() + " Status to " + videoInfo.getStatus());
            } catch (Exception e){
                logger.error(String.format("Error in Updating Clustering Result from Kafka. Error: %s", e));
            }
        }
    }

    private void updateSlowItemsJob(){
        logger.info("Start UpdateSlowItemsJob");
        while (true){
            try{
                List<VideoInfo> videoInfos = getSlowItems();
                videoInfos.forEach(videoInfo -> {
                    try {
                        messageManager.sendVideoInfo(videoInfo);
                        updateVideoSentTime(videoInfo);
                    } catch (JsonProcessingException e) {
                        logger.error(String.format("Error in Sending Slow videoInfo via Kafka. Error: %s", e));
                    }
                });
                logger.info("Sent "+ videoInfos.size() +" Slow Videos " + videoInfos.stream().map(VideoInfo::getId).toList());
                Thread.sleep(oneDayTimeStamp/2);
            } catch (Exception e){
                logger.error(String.format("Error in Sending Slow videoInfo via Kafka. Error: %s", e));
            }
        }
    }

    @PostConstruct
    private void init(){
        Thread saveDataThread = new Thread(this::updateClusteringResultJob);
//        Thread updateSlowItemsThread = new Thread(this::updateSlowItemsJob);
        saveDataThread.start();
//        updateSlowItemsThread.start();
    }

}
