package com.tmt.videoCluster.Config;


import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.config.AbstractMongoClientConfiguration;


@Configuration
public class MongoDBConfig extends AbstractMongoClientConfiguration {

    @Autowired
    private Config config;

    @Override
    protected String getDatabaseName() {
        return config.mongo.getDatabase();
    }

    @Override
    public MongoClient mongoClient() {
        String str = String.format("mongodb://%s:%s@%s:%s",
                config.mongo.getUsername(), config.mongo.getPassword(), config.mongo.getHost(), config.mongo.getPort());
        System.out.println(str);
        ConnectionString connectionString = new ConnectionString(str);
        MongoClientSettings mongoClientSettings = MongoClientSettings.builder()
                .applyConnectionString(connectionString)
                .build();

        return MongoClients.create(mongoClientSettings);
    }
}