// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.jdisc.http.server.jetty;

import com.google.inject.AbstractModule;
import com.google.inject.Module;
import com.google.inject.util.Modules;
import com.yahoo.jdisc.application.BindingRepository;
import com.yahoo.jdisc.handler.RequestHandler;
import com.yahoo.jdisc.http.ConnectorConfig;
import com.yahoo.jdisc.http.ServerConfig;
import com.yahoo.jdisc.http.ServletPathsConfig;
import com.yahoo.jdisc.http.filter.RequestFilter;
import com.yahoo.jdisc.http.filter.ResponseFilter;
import com.yahoo.jdisc.http.guiceModules.ConnectorFactoryRegistryModule;
import com.yahoo.jdisc.http.guiceModules.ServletModule;
import com.yahoo.jdisc.http.server.FilterBindings;

import java.io.IOException;
import java.nio.file.Path;

/**
 * @author Simon Thoresen Hult
 * @author bjorncs
 */
public class TestDrivers {

    public static TestDriver newConfiguredInstance(RequestHandler requestHandler,
                                                   ServerConfig.Builder serverConfig,
                                                   ConnectorConfig.Builder connectorConfig,
                                                   Module... guiceModules) throws IOException {
        return TestDriver.newInstance(
                JettyHttpServer.class,
                requestHandler,
                newConfigModule(serverConfig, connectorConfig, guiceModules));
    }

    public static TestDriver newInstance(RequestHandler requestHandler, Module... guiceModules) throws IOException {
        return TestDriver.newInstance(
                JettyHttpServer.class,
                requestHandler,
                newConfigModule(
                        new ServerConfig.Builder(),
                        new ConnectorConfig.Builder(),
                        guiceModules
                ));
    }

    public enum TlsClientAuth { NEED, WANT }

    public static TestDriver newInstanceWithSsl(RequestHandler requestHandler,
                                                Path certificateFile,
                                                Path privateKeyFile,
                                                TlsClientAuth tlsClientAuth,
                                                Module... guiceModules) throws IOException {
        return TestDriver.newInstance(
                JettyHttpServer.class,
                requestHandler,
                newConfigModule(
                        new ServerConfig.Builder(),
                        new ConnectorConfig.Builder()
                                .tlsClientAuthEnforcer(
                                        new ConnectorConfig.TlsClientAuthEnforcer.Builder()
                                                .enable(true)
                                                .pathWhitelist("/status.html"))
                                .ssl(new ConnectorConfig.Ssl.Builder()
                                             .enabled(true)
                                             .clientAuth(tlsClientAuth == TlsClientAuth.NEED
                                                                 ? ConnectorConfig.Ssl.ClientAuth.Enum.NEED_AUTH
                                                                 : ConnectorConfig.Ssl.ClientAuth.Enum.WANT_AUTH)
                                             .privateKeyFile(privateKeyFile.toString())
                                             .certificateFile(certificateFile.toString())
                                             .caCertificateFile(certificateFile.toString())),
                        Modules.combine(guiceModules)));
    }

    private static Module newConfigModule(ServerConfig.Builder serverConfig,
                                          ConnectorConfig.Builder connectorConfigBuilder,
                                          Module... guiceModules) {
        return Modules.combine(
                new AbstractModule() {
                    @Override
                    protected void configure() {
                        bind(ServletPathsConfig.class).toInstance(new ServletPathsConfig(new ServletPathsConfig.Builder()));
                        bind(ServerConfig.class).toInstance(new ServerConfig(serverConfig));
                        bind(ConnectorConfig.class).toInstance(new ConnectorConfig(connectorConfigBuilder));
                        bind(FilterBindings.class).toInstance(
                                new FilterBindings(
                                        new BindingRepository<>(),
                                        new BindingRepository<>()));
                    }
                },
                new ConnectorFactoryRegistryModule(connectorConfigBuilder),
                new ServletModule(),
                Modules.combine(guiceModules));
    }

}
