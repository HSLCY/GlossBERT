import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


/**
 * This is a modification of the scorer of the SemEval 2015 task 13 on 
 * Multilingual All-Words Sense Disambiguation and Entity Linking.
 *
 */
public class Scorer {

	public static void main(String[] args) throws IOException {
		// Check command-line arguments.
		if (args.length != 2) {
			exit();
		}

		// Load gold standard and system annotations.
		File gs = new File(args[0]);
		if (!gs.exists()) exit();
		File system = new File(args[1]);
		if (!system.exists()) exit();

		// Compute measures.
		Double[] m = score(gs, system);
		System.out.println("P=\t"+String.format("%.1f", m[0]*100)+"%");
		System.out.println("R=\t"+String.format("%.1f", m[1]*100)+"%");
		System.out.println("F1=\t"+String.format("%.1f", m[2]*100)+"%");
	}
	
	private static void exit() {
		System.out.println("Scorer gold-standard_key_file system_key_file\n");
		System.exit(0);
	}
	
	public static Double[] score(File gs, File system) throws IOException {
		// Read the input files.
		Map<String, Set<String>> gsMap = new HashMap<String, Set<String>>();
		readFile(gs, gsMap);
		Map<String, Set<String>> systemMap = new HashMap<String, Set<String>>();
		readFile(system, systemMap);
		// Count how many good and bad answers the system gives.
		double ok = 0, notok = 0;
		for (String key : systemMap.keySet()) {
			// If the fragment of text annotated by the system is not contained in the gold
			// standard then skip it.
			if (!gsMap.containsKey(key)) continue;
			// Handling multiple answers for a same fragment of text.
			int local_ok = 0, local_notok = 0;
			for (String answer : systemMap.get(key)) {
				if (gsMap.get(key).contains(answer)) local_ok++;
				else local_notok++;
			}
			ok += local_ok/(double)systemMap.get(key).size();
			notok += local_notok/(double)systemMap.get(key).size();
		}
		// Compute precision, recall and f1 scores.
		Double[] m = new Double[3];
		m[0] = ok/(double)(ok+notok);
		m[1] = ok/(double)gsMap.size();
		if (m[0]+m[1]==0.0) {m[2]=0.0;}
		else {m[2] = (2*m[0]*m[1]) / (m[0]+m[1]);}
		return m;
	}
	
	public static void readFile(File file, Map<String, Set<String>> map) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(file));
		String l;
		int cnt=0;
		while ((l = in.readLine()) != null) 
		{
			cnt++;
			String[] ll = l.split(" ");
			if (ll.length<2){
			   System.out.println("line number "+cnt+" not complete: "+l);
			   continue;
			}	
			// Update the map with a new set of answers.
			if (!map.containsKey(ll[0])) map.put(ll[0], new HashSet<String>()); 
			for (int i = 1; i < ll.length; i++) map.get(ll[0]).add(ll[i]);
		
		}
		in.close();
	}
}
