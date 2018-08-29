package de.upb.crc901.proseco;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

/**
 * 
 * PrototypeProperties, given a property file, PrototypeProperties reads the
 * file and stores properties as key-value( or value list) pairs.
 * 
 * Property file should have the following format:
 * 
 * <ol>
 * <li>Simple Key-Value pair: "key=value"</li>
 * <li>Key-Value list pair: "key=value1,value2,..."</li>
 * <li>Comment: "#commented line"</li>
 * </ol>
 *
 */
public class PrototypeProperties extends HashMap<String, List<String>> {

	/**
	 *
	 */
	private static final long serialVersionUID = 5554263929565731068L;

	// public static final String K_DATAZIP = "datazip";
	// public static final String K_INSTANCES_SERIALIZED =
	// "serializedInstances";

	/**
	 * PrototypeProperties constructor with file path as String
	 * 
	 * @param propertiesFile,
	 *            path of properties file as String
	 */
	public PrototypeProperties(final String propertiesFile) {
		this(new File(propertiesFile));
	}

	/**
	 * PrototypeProperties constructor with file
	 * 
	 * @param propertiesFile,
	 *            properties file
	 */
	public PrototypeProperties(final File propertiesFile) {
		try (BufferedReader br = new BufferedReader(new FileReader(propertiesFile))) {
			String line;
			while ((line = br.readLine()) != null) {
				if (line.trim().startsWith("#")) {
					continue;
				}

				if (line.contains("=")) {
					final String[] lineSplit = line.split("=");
					if (lineSplit.length != 2) {
						System.out.println("WARN: Ignored malformed properties line: " + line);
						continue;
					}

					final List<String> properties = new LinkedList<>();
					if (lineSplit[1].contains(",")) {
						final String[] listElements = lineSplit[1].split(",");
						for (final String elem : listElements) {
							properties.add(elem.trim());
						}
					} else {
						properties.add(lineSplit[1].trim());
					}

					this.put(lineSplit[0].trim(), properties);
				}
			}
		} catch (final FileNotFoundException e) {
			e.printStackTrace();
		} catch (final IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * getProperty, returns value of the property with the given key
	 * 
	 * @param key
	 *            key of the property
	 * @return value of the property
	 */
	public String getProperty(final String key) {
		if (!this.containsKey(key)) {
			return null;
		} else if (this.get(key).size() == 0) {
			return null;
		}

		return this.get(key).get(0);
	}

	/**
	 * getPropertyList, returns property list with the given key
	 * 
	 * @param key
	 *            key of the property list
	 * @return values of property list
	 */
	public List<String> getPropertyList(final String key) {
		return this.get(key);
	}

}
