package org.spark.clustering.coefficient;

import java.util.*;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class ClusCoe {
    private static final Pattern SPACE = Pattern.compile(" ");

    public static JavaPairRDD<String, Double> compute(JavaRDD<String> lines) {

        JavaPairRDD<Long, Long> nodePairRddFirst = lines.flatMapToPair((String s) -> {
            String x[] = SPACE.split(s);
            long source = Long.parseLong(x[0]);
            long target = Long.parseLong(x[1]);

            return Arrays.asList(new Tuple2<>(source, target), new Tuple2<>(target, source)).iterator();
        });
        nodePairRddFirst.cache();
        nodePairRddFirst.saveAsTextFile("/home/vagrant/nodePairRddFirst");


        /**
         * Δημιουργία RDD με τους γειτονικούς κόμβους
         * Δημιουργία RDD με όλους τους δυνατούς συνδιασμούς ζευγαριών των γειτόνων για κάθε κόμβο
         */

        JavaPairRDD<Long, Iterable<Long>> adjacencyList = nodePairRddFirst.groupByKey();
        adjacencyList.cache();
        adjacencyList.saveAsTextFile("/home/vagrant/adjacencyList");


        JavaPairRDD<Tuple2<Long, Long>, Long> pCombinationPair = adjacencyList.flatMapToPair(s -> {

            List<Tuple2<Tuple2<Long, Long>, Long>> listCombinationPair = new ArrayList<>();

            Long nodeM = s._1;
            for (long nodeF : s._2()) {
                for (long nodeS : s._2()) {

                    if (nodeF != nodeS) {
                        if (nodeS > nodeF) {
                            listCombinationPair.add(new Tuple2<>(new Tuple2<>(nodeF, nodeS), nodeM));
                        }
                    }
                }
            }
            return listCombinationPair.iterator();
        });
        pCombinationPair.cache();
        pCombinationPair.saveAsTextFile("/home/vagrant/pCombinationPair");

        /**
         * Δημιουργία RDD με τον "βαθμό κόμβου" για κάθε κόμβο
         [Υπολογισμός παρανομαστή του κλάσματος του μέσου όρου των συντελεστών ομαδοποίησης των κόμβων]
         **/
        JavaPairRDD<Long, Double> calcuNodeDegreeDen = nodePairRddFirst.mapToPair(s -> {

            return new Tuple2<>(s._2, 1);

        }).reduceByKey((a, b) -> {
            return a + b;

        }).mapToPair((s) -> {

            Double s2 = new Double(s._2);
            Long s1 = s._1;

            Double calcDegree = (s2 * (s2 - 1)) / 2;

            return new Tuple2<>(s1, calcDegree);
        });
        calcuNodeDegreeDen.cache();

        calcuNodeDegreeDen.saveAsTextFile("/home/vagrant/calcuNodeDegreeDen");

        /**
         * Μετασχηματισμός του πρώτου RDD στην μορφή Tuple2<Long, Long>, Long για να γίνει join.(nodePairRddT)
         * join RDDs για να βρώ τα γειτονικά node που είναι συνδεδεμένα μεταξύ τους με ακμή.(nodeNeighPointSum)
         * Υπολογισμός Clustering Coefficient για κάθε node και Global Clustering Coefficient των κόμβων].
         * (unClusterCoeffNode, globalClusCoeff)
         */

        JavaPairRDD<Tuple2<Long, Long>, Long> nodePairRddT = nodePairRddFirst.mapToPair(s -> {

            return new Tuple2<>(new Tuple2<>(s._1, s._2), 0L);
        });
        nodePairRddT.cache();

        JavaPairRDD<Long, Double> nodeNeighPointSum =
                pCombinationPair.join(nodePairRddT).mapToPair(s -> {

                    return new Tuple2<>(s._2._1, 1.0);
                }).reduceByKey((a, b) -> {
                    return a + b;
                });
        nodeNeighPointSum.cache();
        nodeNeighPointSum.saveAsTextFile("/home/vagrant/nodeNeighPointSum");

        JavaPairRDD<Long, Double> unClusterCoeffNode =
                nodeNeighPointSum.join(calcuNodeDegreeDen).mapToPair(s -> {

                    Double sum = s._2._1 / s._2._2;

                    return new Tuple2<>(s._1, sum);
                });

        unClusterCoeffNode.cache();
        //Μέγεθος κόμβων

        Long sizeOfRank = adjacencyList.count();
        Double sizeOfRankD = Double.parseDouble(sizeOfRank.toString());

        JavaPairRDD<String, Double> globalClusCoeff = unClusterCoeffNode.mapToPair(s -> {

            return new Tuple2<>("all_Nodes", s._2);
        }).reduceByKey((a, b) -> {
            return (a + b);
        }).mapToPair(s -> {

            Double globalSum = s._2 / sizeOfRankD;

            return new Tuple2<>("GlobalClusteringCoefficient --> ", globalSum);
        });
        globalClusCoeff.cache();
        unClusterCoeffNode.saveAsTextFile("/home/vagrant/unClusterCoeffNode");

        globalClusCoeff.saveAsTextFile("/home/vagrant/globalClusCoeff");

        return globalClusCoeff;
    }


    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            System.err.println("Usage: Fof <input-path> <output-path>");
            System.exit(1);
        }

        SparkConf sparkConf = new SparkConf().setAppName("ClusCoe");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> lines = sc.textFile(args[0]);

        JavaPairRDD<String, Double> clusCoeEdges = compute(lines);

        clusCoeEdges.saveAsTextFile(args[1]);

        sc.stop();

    }
}