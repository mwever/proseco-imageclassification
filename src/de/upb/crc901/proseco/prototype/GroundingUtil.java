package de.upb.crc901.proseco.prototype;

public class GroundingUtil {

  public static String[] compileJava(final String fileToCompile, final String classPathDependencies) {
    String[] processCommand;

    if (classPathDependencies == null || classPathDependencies.equals("")) {
      processCommand = new String[2];
    } else {
      processCommand = new String[4];
    }

    processCommand[0] = "javac";
    processCommand[processCommand.length - 1] = fileToCompile;

    if (processCommand.length > 2) {
      processCommand[1] = "-cp";
      processCommand[2] = "\"" + classPathDependencies + "\"";
    }

    StringBuilder sb = new StringBuilder();
    for (String commandPart : processCommand) {
      sb.append(commandPart + " ");
    }
    System.out.println(sb.toString());

    return processCommand;
  }

  public static String[] compileJava(final String fileToCompile) {
    return compileJava(fileToCompile, null);
  }

  public static String[] executeJava(final String classToExecute, final String params, final String classPathDependencies) {
    String[] processCommand = new String[2];

    String[] paramsArray = params.split(" ");
    if (classPathDependencies == null || classPathDependencies.equals("")) {
      processCommand = new String[2];
    } else {
      processCommand = new String[4 + paramsArray.length];
    }

    int commandIndex = 0;
    processCommand[commandIndex++] = "java";

    if (processCommand.length > 2) {
      processCommand[commandIndex++] = "-cp";
      processCommand[commandIndex++] = "\"" + classPathDependencies + "\"";
    }

    processCommand[commandIndex++] = classToExecute;

    for (String param : paramsArray) {
      paramsArray[commandIndex++] = param;
    }

    return processCommand;
  }

  public static String[] executeJava(final String classToExecute) {
    return executeJava(classToExecute, "", null);
  }

}
