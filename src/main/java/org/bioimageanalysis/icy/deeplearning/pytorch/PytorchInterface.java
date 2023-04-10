package org.bioimageanalysis.icy.deeplearning.pytorch;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.bioimageanalysis.icy.deeplearning.engine.DeepLearningEngineInterface;
import org.bioimageanalysis.icy.deeplearning.exceptions.LoadModelException;
import org.bioimageanalysis.icy.deeplearning.exceptions.RunModelException;
import org.bioimageanalysis.icy.deeplearning.pytorch.tensor.ImgLib2Builder;
import org.bioimageanalysis.icy.deeplearning.pytorch.tensor.TensorBuilder;
import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.pytorch.CompilationUnit;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.JitModule;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch.DeviceType;
import org.bytedeco.pytorch.presets.torch;

import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;


/**
 * This class implements an interface that allows the main plugin
 * to interact in an agnostic way with the Deep Learning model and
 * tensors to make inference.
 * This implementation add the Pytorch support to the main program.
 * 
 * @see SequenceBuilder SequenceBuilder: Create sequences from tensors.
 * @see TensorBuilder TensorBuilder: Create tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro 
 */
public class PytorchInterface implements DeepLearningEngineInterface
{
	private Object model;
    
    public PytorchInterface()
    {    
    }
    
    public static void main(String[] args) {
    	/* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");
    	// Load a Torchscript model
    	String modelSource = "/Users/runner/work/macos-test-2/macos-test-2/weights-torchscript.pt";
    	BytePointer filenamePointer = new BytePointer(modelSource);
    	//JitModule model = org.bytedeco.pytorch.global.torch.import_ir_module(new CompilationUnit(),
    		//	modelSource);
    	JitModule model = org.bytedeco.pytorch.global.torch.load(filenamePointer);
    	//org.bytedeco.pytorch.global.torch.ExportModule(model, filenamePointer);
    	
    	// Create an input tensor
    	org.bytedeco.pytorch.Tensor inputTensor = 
    			org.bytedeco.pytorch.Tensor.create(new float[256* 256], 
    											new long[] {1, 1, 256, 256});
    	
    	// Run the model
    	boolean aa = model.is_training();
		model.eval();
    	boolean bb = model.is_training();
    	IValue outputTensor = model.forward(new IValueVector(new IValue(inputTensor)));
    	
    	// Print shape of the output tensor
    	System.out.println(Arrays.toString(outputTensor.toTensor().shape()));
    	
    	// Retrieve result
    	float[] array = new float[256 * 256 * 8];
    	Indexer indexer = outputTensor.toTensor().createIndexer();
    	double dd = indexer.getDouble(new long[] {0, 0, 0, 0});
    	double ddd = indexer.getDouble(new long[] {0, 1, 0, 0});
    	System.out.println(dd);
    	System.out.println(ddd);
    	System.out.println("Done");
    }

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		List<org.bytedeco.pytorch.Tensor> inputList = new ArrayList<org.bytedeco.pytorch.Tensor>();
        List<String> inputListNames = new ArrayList<String>();
        for (Tensor<?> tt : inputTensors) {
        	inputListNames.add(tt.getName());
        	inputList.add(TensorBuilder.build(tt));
        }
        // Run model
        List<org.bytedeco.pytorch.Tensor> outputNDArrays = null;
		// Fill the agnostic output tensors list with data from the inference result
		outputTensors = fillOutputTensors(outputNDArrays, outputTensors);
	}

	@Override
	public void loadModel(String modelFolder, String modelSource) throws LoadModelException {
		String modelName = new File(modelSource).getName();
		modelName = modelName.substring(0, modelName.indexOf(".pt"));
		try {
		} catch (Exception e) {
			e.printStackTrace();
			managePytorchExceptions(e);
			throw new LoadModelException("Error loading a Pytorch model", e.getCause().toString());
		}
	}

	@Override
	public void closeModel() {
		if (model != null)
		model = null;		
	}
	
	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning engine
	 * that can be readable by Deep Icy
	 * @param outputTensors
	 * 	an NDList containing NDArrays (tensors)
	 * @param outputTensors2
	 * 	the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same as the number of
	 * 	Tensors outputed by the model
	 */
	public static List<Tensor<?>> fillOutputTensors(List<org.bytedeco.pytorch.Tensor> outputNDArrays, List<Tensor<?>> outputTensors) throws RunModelException{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i ++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
		return outputTensors;
	}
	
	/**
	 * Print the correct message depending on the exception produced when
	 * trying to load the model
	 * 
	 * @param ex
	 * 	the exception that occurred
	 */
	public static void managePytorchExceptions(Exception e) {
		if (e instanceof MalformedURLException) {
			System.out.println("No model was found in the folder provided.");
		} else if (e instanceof Exception) {
			String err = e.getMessage();
			String os = System.getProperty("os.name").toLowerCase();
			String msg;
			if (os.contains("win") && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")) {
				msg = "DeepIcy could not load the model.\n" + 
					"Please install the Visual Studio 2019 redistributables and reboot" +
					"your machine to be able to use Pytorch with DeepIcy.\n" +
					"For more information:\n" +
					" -https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md\n" +
					" -https://github.com/awslabs/djl/issues/126\n" +
					"If you already have installed VS2019 redistributables, the error" +
					"might be caused by a missing dependency or an incompatible Pytorch version.\n" + 
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto).\n" +
					"should be compatible with each other." +
					"Please check the DeepIcy Wiki.";
			} else if((os.contains("linux") || os.contains("unix")) && err.contains("https://github.com/awslabs/djl/blob/master/docs/development/troubleshooting.md")){
				msg  = "DeepIcy could not load the model.\n" +
					"Check that there are no repeated dependencies on the jars folder.\n" +
					"The problem might be caused by a missing or repeated dependency or an incompatible Pytorch version.\n" +
					"Furthermore, the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " +
					"should be compatible with each other.\n" +
					"If the problem persists, please check the DeepIcy Wiki.";
			} else {
				msg  = "DeepIcy could not load the model.\n" +
					"Either the DJL Pytorch version is incompatible with the Torchscript model's " +
					"Pytorch version or the DJL Pytorch dependencies (pytorch-egine, pytorch-api and pytorch-native-auto) " + 
					"are not compatible with each other.\n" +
					"Please check the DeepIcy Wiki.";
			}
			System.out.println(msg);
		} else if (e instanceof IOException) {
			String msg = "DeepImageJ could not load the model.\n" + 
				"The model provided is not a correct Torchscript model.";
			System.out.println(msg);
		} else if (e instanceof IOException) {
			System.out.println("An error occurred accessing the model file.");
		}
	}
	
	public void finalize() {
		System.out.println("Collected Garbage");
	}
}
