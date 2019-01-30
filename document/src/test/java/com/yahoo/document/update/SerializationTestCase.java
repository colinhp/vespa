// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.document.update;

import com.yahoo.document.*;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.document.serialization.*;
import com.yahoo.io.GrowableByteBuffer;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import org.junit.Before;
import org.junit.Test;

import java.io.FileOutputStream;

import static org.junit.Assert.assertEquals;

/**
 * @author bratseth
 */
public class SerializationTestCase {

    private DocumentType documentType;

    private Field field;
    private final static TensorType tensorType = new TensorType.Builder().mapped("x").mapped("y").build();
    private Field tensorField;

    @Before
    public void setUp() {
        documentType = new DocumentType("document1");
        field = new Field("field1", DataType.getArray(DataType.STRING));
        documentType.addField(field);
        tensorField = new Field("tensorfield", new TensorDataType(tensorType));
        documentType.addField(tensorField);
    }

    @Test
    public void testAddSerialization() {
        FieldUpdate update = FieldUpdate.createAdd(field, new StringFieldValue("value1"));
        DocumentSerializer buffer = DocumentSerializerFactory.create6();
        update.serialize(buffer);

        buffer.getBuf().rewind();

        try{
            FileOutputStream fos = new FileOutputStream("src/test/files/addfieldser.dat");
            fos.write(buffer.getBuf().array(), 0, buffer.getBuf().remaining());
            fos.close();
        } catch (Exception e) {}

        FieldUpdate deserializedUpdate = new FieldUpdate(DocumentDeserializerFactory.create6(new DocumentTypeManager(), buffer.getBuf()), documentType, Document.SERIALIZED_VERSION);
        assertEquals("'field1' [add value1 1]", deserializedUpdate.toString());
    }

    @Test
    public void testClearSerialization() {
        FieldUpdate update = FieldUpdate.createClear(field);
        DocumentSerializer buffer = DocumentSerializerFactory.create6();
        update.serialize(buffer);

        buffer.getBuf().rewind();
        FieldUpdate deserializedUpdate = new FieldUpdate(DocumentDeserializerFactory.create6(new DocumentTypeManager(), buffer.getBuf()), documentType, Document.SERIALIZED_VERSION);

        assertEquals("'field1' [clear]", deserializedUpdate.toString());
    }

    private static TensorFieldValue createTensor(TensorType type, String tensorCellString) {
        return new TensorFieldValue(Tensor.from(type, tensorCellString));
    }

    private GrowableByteBuffer serializeUpdate(FieldUpdate update) {
        DocumentSerializer buffer = DocumentSerializerFactory.createHead(new GrowableByteBuffer());
        update.serialize(buffer);
        buffer.getBuf().rewind();
        return buffer.getBuf();
    }

    private FieldUpdate deserializeUpdate(GrowableByteBuffer buffer) {
        return new FieldUpdate(DocumentDeserializerFactory.createHead(new DocumentTypeManager(), buffer), documentType, Document.SERIALIZED_VERSION);
    }

    @Test
    public void testTensorModifySerialization() {
        FieldUpdate update = new FieldUpdate(tensorField);
        TensorFieldValue tensor = createTensor(tensorType, "{{x:8,y:9}:2}");
        update.addValueUpdate(new TensorModifyUpdate(TensorModifyUpdate.Operation.REPLACE, tensor));
        GrowableByteBuffer buffer = serializeUpdate(update);
        FieldUpdate deserializedUpdate = deserializeUpdate(buffer);
        assertEquals("tensorfield", deserializedUpdate.getField().getName());
        assertEquals(1, deserializedUpdate.getValueUpdates().size());
        ValueUpdate valueUpdate = deserializedUpdate.getValueUpdate(0);
        if (!(valueUpdate instanceof TensorModifyUpdate)) {
            throw new IllegalStateException("Expected tensorModifyUpdate");
        }
        TensorModifyUpdate tensorModifyUpdate = (TensorModifyUpdate) valueUpdate;
        assertEquals(TensorModifyUpdate.Operation.REPLACE, tensorModifyUpdate.getOperation());
        assertEquals(tensor, tensorModifyUpdate.getValue());
        assertEquals(update, deserializedUpdate);
    }
}
