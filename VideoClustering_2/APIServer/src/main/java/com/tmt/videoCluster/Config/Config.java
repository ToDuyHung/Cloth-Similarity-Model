package com.tmt.videoCluster.Config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@ConfigurationProperties()
@Data
public class Config {

    @Data
    public static class MongoConfig {
        private String host;
        private String port;
        private String username;
        private String password;
        private String database;
    }

    @Data
    public static class KafkaConfig {
        private String bootstrapAddress;
        private String videoLinkTopic;
        private String clusterResultTopic;
        private String groupId;
    }

    MongoConfig mongo;
    KafkaConfig kafka;
}