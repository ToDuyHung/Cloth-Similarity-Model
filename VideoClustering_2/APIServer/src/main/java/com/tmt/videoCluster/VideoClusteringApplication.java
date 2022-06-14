package com.tmt.videoCluster;

import com.tmt.videoCluster.Config.Config;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication
@EnableConfigurationProperties({Config.class})
public class VideoClusteringApplication {
	public static void main(String[] args) {
		SpringApplication.run(VideoClusteringApplication.class, args);
	}
}
