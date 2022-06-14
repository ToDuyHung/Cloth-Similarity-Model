package com.tmt.videoCluster.Services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.tmt.videoCluster.Config.KafkaTopicConfig;
import com.tmt.videoCluster.Models.VideoInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.stereotype.Service;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.util.concurrent.ListenableFutureCallback;

import java.util.concurrent.LinkedBlockingQueue;


@Service
public class MessageManager {
    private final static Logger logger = LoggerFactory.getLogger(VideoDataManager.class.getName());

    @Autowired
    KafkaTopicConfig kafkaTopicConfig;

    @Autowired
    KafkaTemplate<String, String> kafkaTemplate;

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static final LinkedBlockingQueue<VideoInfo> messageQueue = new LinkedBlockingQueue<>();

    public void sendVideoInfo(VideoInfo message) throws JsonProcessingException {
        String messageString = objectMapper.writeValueAsString(message);
        ListenableFuture<SendResult<String, String>> future =
                kafkaTemplate.send(kafkaTopicConfig.getPublishTopic(), messageString);

        future.addCallback(new ListenableFutureCallback<>() {
            @Override
            public void onSuccess(SendResult<String, String> result) {
                logger.info("Sent message=[" + messageString + "] with offset=[" + result.getRecordMetadata().offset() + "]");
            }

            @Override
            public void onFailure(Throwable e) {
                logger.info("Unable to send message=[" + messageString + "] due to : " + e.getMessage());
            }
        });
    }

    @KafkaListener(topics = "${kafka.clusterResultTopic}" , groupId = "${kafka.groupId}", containerFactory = "kafkaListenerContainerFactory")
    public void listenGroupFoo(String message) {
        try {
            VideoInfo videoInfo = objectMapper.readValue(message, VideoInfo.class);
            logger.info("Receive cluster info of video" + videoInfo.getId());
            messageQueue.put(videoInfo);
        } catch (JsonProcessingException | InterruptedException e){
            logger.error(String.format("Error: %s", e));
        }
    }

    public VideoInfo getClusteringResult() {
        try {
            return messageQueue.take();
        } catch (InterruptedException e){
            logger.error(String.format("Cannot get result from buffer. Error: %s", e));
            return null;
        }
    }
}
