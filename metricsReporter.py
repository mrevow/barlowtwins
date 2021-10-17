from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

class MetricsReporter(object):
  def __init__(self, args, logger):
    self.args = args
    self.logger = logger

  def atRecall(self, prec, rec, thresh, atRecall=0.75):
    at_rec = [(p, r, t) for p, r, t in zip(prec, rec, thresh) if r <= atRecall]
    return at_rec[0] if len(at_rec) > 0 else (-1, -1, -1)

  def atPrec(self, prec, rec, thresh, atPrec=0.75):
    at_prec = [(p, r, t) for p, r, t in zip(prec, rec, thresh) if p >= atPrec]
    return at_prec[0] if len(at_prec) > 0 else (-1, -1, -1)

  def atFpr(self, fpr, tpr, thresh, atFpr=0.75):
    at_fpr = [(fp, tp, t) for idx, (fp, tp, t) in enumerate(zip(fpr, tpr, thresh)) if fp >= atFpr]
    return at_fpr[0] if len(at_fpr) > 0 else (-1, -1, -1)

  def atTpr(self, fpr, tpr, thresh, atTpr=0.75):
    at_tpr = [(fp, tp, t) for idx, (fp, tp, t) in enumerate(zip(fpr, tpr, thresh)) if tp >= atTpr]
    return at_tpr[0] if len(at_tpr) > 0 else (-1, -1, -1)   

  def calcBinaryStats(self, preds, labels, name='', doPlotCurves=False):
    stats = None
    if len(preds) == len(labels) and len(preds) > 0:
      stats = {}
      precision, recall, thresholds = precision_recall_curve(labels, preds)
      stats.update({"auc": auc(recall, precision)})
      stats.update({"pr@_{}".format(r): "{:.3f}".format(self.atRecall(precision, recall, thresholds, atRecall=r)[0]) for r in self.args.music_classifier_metrics_at_recall})
      stats.update({"rec@_{}".format(p): "{:.3f}".format(self.atPrec(precision, recall, thresholds, atPrec=p)[1]) for p in self.args.music_classifier_metrics_at_precision})


      fpr, tpr, rocThresholds = roc_curve(labels, preds)
      stats.update({"roc": auc(fpr, tpr)})
      stats.update({"tpr@_{}".format(fp): "{:.3f}".format(self.atFpr(fpr, tpr, rocThresholds, atFpr=fp)[1]) for fp in self.args.music_classifier_metrics_at_fpr})
      stats.update({"fpr@_{}".format(tp): "{:.3f}".format(self.atTpr(fpr, tpr, rocThresholds, atTpr=tp)[0]) for tp in self.args.music_classifier_metrics_at_tpr})

      if doPlotCurves:
        self.plotPrCurve(name, "PR AUC", recall, precision,)
        self.plotPrCurve(name, "ROC AUC", fpr, tpr, xLabel='FPR', yLabel='TPR')

    return stats

  def plotStats(self, stats, ite, typ):
    if self.args.rank == 0 and stats is not None:
      for k, v in stats.items():
        try:
          val = float(v)
          nme = "{}_{}".format(typ, k)
          maxx =  self.args.data_plot_max_limits.get(k, None)
          val = min(val, maxx) if maxx is not None else val
          minn =  self.args.data_plot_min_limits.get(k, None)
          val = max(val, minn) if minn is not None else val
          self.logger.log_row(name=nme, iter=ite, val=val, description="{} master proc".format(nme))
        except ValueError:
          pass

  def plotPrCurve(self, name, nameDetail, xData, yData, xLabel='Precision', yLabel='Recall'):
    plt.close('all')
    plt.figure(figsize=(13,7))
    fig, ax = plt.subplots()
    plt.plot(xData, yData)
    title = '{} {} = {:.2f}'.format(name, nameDetail, metrics.auc( xData, yData),)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if self.args.music_classifier_metrics_plt_pause_delay_sec > 0:
      plt.pause(self.args.music_classifier_metrics_plt_pause_delay_sec)
    self.logger.log_image(title,  plot=fig)   

  def logValues(self, stats, typ):
    for k, v in stats.items():
      nme = "{}_{}".format(typ, k)
      self.logger.log_value(name=nme, value=v)

  