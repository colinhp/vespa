// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.config.server.maintenance;

import com.yahoo.log.LogLevel;
import com.yahoo.vespa.config.server.ApplicationRepository;
import com.yahoo.vespa.curator.Curator;
import com.yahoo.vespa.flags.FlagSource;

import java.time.Duration;

/**
 * Removes expired sessions and locks
 * <p>
 * Note: Unit test is in ApplicationRepositoryTest
 *
 * @author hmusum
 */
public class SessionsMaintainer extends ConfigServerMaintainer {
    private final boolean hostedVespa;

    SessionsMaintainer(ApplicationRepository applicationRepository, Curator curator, Duration interval, FlagSource flagSource) {
        super(applicationRepository, curator, flagSource, Duration.ofMinutes(1), interval);
        this.hostedVespa = applicationRepository.configserverConfig().hostedVespa();
    }

    @Override
    protected boolean maintain() {
        applicationRepository.deleteExpiredLocalSessions();

        if (hostedVespa) {
            Duration expiryTime = Duration.ofHours(2);
            int deleted = applicationRepository.deleteExpiredRemoteSessions(expiryTime);
            log.log(LogLevel.FINE, () -> "Deleted " + deleted + " expired remote sessions older than " + expiryTime);
        }

        Duration lockExpiryTime = Duration.ofHours(2);
        int deleted = applicationRepository.deleteExpiredSessionLocks(lockExpiryTime);
        log.log(LogLevel.FINE, () -> "Deleted " + deleted + " locks older than " + lockExpiryTime);

        return true;
    }

}
