package com.tmt.videoCluster.Models;


import lombok.Data;
import org.springframework.data.mongodb.core.mapping.Document;

import java.util.Collection;
import java.util.Date;
import java.util.List;

@Document(collection = "Video")
@Data
public class VideoInfo {

    public enum Status {
        CREATED,
        PROCESSING,
        FAILED,
        DONE
    }
    @Data
    public static class ClusterInfo{
        private String id;
        private int begin;
        private int end;
        private String imageUrl;
        private String productId;
    }
    private String id;
    private String videoUrl;
    private long samplingTime;
    private long createdTime = new Date().getTime();
    private long sentTime;
    private long doneTime;
    private Status status = Status.CREATED;
    private List<ClusterInfo> clusterInfos;
    public boolean isDone(){
        return status == Status.DONE;
    }
    public void updateSentTime(){
        sentTime = new Date().getTime();
    }
}
